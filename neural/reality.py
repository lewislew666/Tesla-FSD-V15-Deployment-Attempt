"""Independent research architecture, with a DINOv3 frontend. No Tesla weights."""
import math
import sys
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel


def position(length, width, device):
    i = torch.arange(length, device=device)[:, None]
    f = torch.exp(torch.arange(0, width, 2, device=device) * (-math.log(10000.)/width))
    return torch.stack(((i*f).sin(), (i*f).cos()), dim=-1).flatten(1)


class MetaBackbone(nn.Module):
    """Expose the patch tokens of Meta's native implementation to SharedVision."""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = SimpleNamespace(hidden_size=model.embed_dim, patch_size=16)

    def forward(self, pixel_values):
        features = self.model.forward_features(pixel_values)
        return SimpleNamespace(last_hidden_state=features["x_norm_patchtokens"])


class SharedVision(nn.Module):
    def __init__(self, encoder, width=96, max_views=8):
        super().__init__()
        self.encoder = encoder.requires_grad_(False)
        self.project = nn.Linear(encoder.config.hidden_size, width)
        self.view = nn.Embedding(max_views, width)
        self.mix = nn.TransformerEncoderLayer(width, 4, 4*width, dropout=0, batch_first=True)

    @classmethod
    def from_checkpoint(cls, model_id, *, meta_repo=None, **kwargs):
        weights = Path(model_id)
        if weights.suffix == ".pth":
            if meta_repo is None or not (Path(meta_repo)/"hubconf.py").is_file():
                raise ValueError("For a Meta .pth checkpoint, provide --meta-repo pointing to the official dinov3 checkout.")
            architecture = weights.name.split("_pretrain_")[0]
            if architecture not in {"dinov3_vits16","dinov3_vits16plus","dinov3_vitb16","dinov3_vitl16"}:
                raise ValueError("Expected an official DINOv3 S, S+, B, or L checkpoint filename.")
            sys.path.insert(0, str(Path(meta_repo).resolve()))
            try:
                builders = import_module("dinov3.hub.backbones")
                model = getattr(builders, architecture)(pretrained=False)
            finally:
                sys.path.pop(0)
            state = torch.load(weights, map_location="cpu", weights_only=True)
            model.load_state_dict(state, strict=True)
            return cls(MetaBackbone(model), **kwargs)
        encoder = AutoModel.from_pretrained(model_id)
        if encoder.config.model_type != "dinov3_vit":
            raise ValueError("Expected a DINOv3 ViT checkpoint.")
        return cls(encoder, **kwargs)

    def patch_grid(self, x):
        """Dense, frozen features [N,C,H/patch,W/patch] from RGB in [0,1]."""
        _, _, h, w = x.shape
        ps = self.encoder.config.patch_size
        if h % ps or w % ps or h < 2*ps or w < 4*ps:
            raise ValueError("Use patch-aligned images of at least 2 by 4 patches.")
        mean = x.new_tensor([.485,.456,.406])[None,:,None,None]
        std = x.new_tensor([.229,.224,.225])[None,:,None,None]
        self.encoder.eval()
        with torch.no_grad():
            tokens = self.encoder(pixel_values=(x-mean)/std).last_hidden_state
        patches = tokens[:, -(h//ps)*(w//ps):].transpose(1,2)
        return patches.reshape(len(x),-1,h//ps,w//ps)

    def forward(self, rgb):
        b, t, v = rgb.shape[:3]
        grid = self.patch_grid(rgb.flatten(0,2))
        x = self.project(F.adaptive_avg_pool2d(grid,(2,4)).flatten(2).transpose(1,2))
        x = x.reshape(b,t,v,8,-1)
        x = x + position(t,x.size(-1),x.device)[None,:,None,None,:]
        x = x + self.view(torch.arange(v,device=x.device))[None,None,:,None,:]
        x = x + position(8,x.size(-1),x.device)[None,None,None,:,:]
        return self.mix(x.flatten(1,3))


class TokenStack(nn.Module):
    def __init__(self, width, layers, heads):
        super().__init__()
        self.blocks = nn.ModuleList(nn.TransformerEncoderLayer(
            width, heads, 4*width, dropout=0, batch_first=True, norm_first=True
        ) for _ in range(layers))
        self.norm = nn.LayerNorm(width)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class ActionHead(nn.Module):
    """Untrained research output; action convention is a project assumption."""
    def __init__(self, shared, width, layers, heads):
        super().__init__()
        self.project = nn.Linear(shared,width)
        self.stack = TokenStack(width,layers,heads)
        self.output = nn.Linear(width,2)

    def forward(self, features):
        return self.output(self.stack(self.project(features)).mean(1))


class VideoFlow(nn.Module):
    """Conditional interval-average velocity in RGB space for an eight-frame block."""
    def __init__(self, shared, width, layers, heads, patch=8):
        super().__init__()
        self.patch = patch
        self.embed = nn.Conv2d(3,width,patch,stride=patch)
        self.condition = nn.Linear(shared,width)
        self.interval = nn.Sequential(nn.Linear(2,width),nn.SiLU(),nn.Linear(width,width))
        self.stack = TokenStack(width,layers,heads)
        self.output = nn.Linear(width,3*patch*patch)

    def forward(self, z, r, t, features):
        b, frames, _, h, w = z.shape
        p = self.patch
        if frames != 8 or h%p or w%p:
            raise ValueError("Expected eight patch-aligned frames.")
        x = self.embed(z.flatten(0,1)).flatten(2).transpose(1,2).reshape(b,-1,self.embed.out_channels)
        x = x + position(x.shape[1],x.shape[2],x.device) + self.interval(torch.stack([r,t],1))[:,None]
        context = self.condition(features)
        x = self.stack(torch.cat([context,x],1))[:,context.shape[1]:]
        tiles = self.output(x).reshape(b,frames,h//p,w//p,3,p,p)
        return tiles.permute(0,1,4,2,5,3,6).reshape_as(z)

    def generate(self, previous, features, noise):
        # Independent endpoint/noise convention. Inputs and outputs are [0,1] RGB.
        source = previous + .2*noise
        t = torch.ones(len(source),device=source.device,dtype=source.dtype)
        return (source-self(source,0*t,t,features)).clamp(0,1)

    def flow_loss(self, previous, target, features):
        # Straight interpolation; mean-flow identity u = v - (t-r) du/dt.
        # Math attention supports forward-mode differentiation through attention.
        from torch.nn.attention import SDPBackend, sdpa_kernel
        source = previous + .2*torch.randn_like(previous)
        v = source-target
        r,t = torch.rand(2,len(v),device=v.device).sort(dim=0).values
        scale = (slice(None),None,None,None,None)
        z = target + t[scale]*v
        with sdpa_kernel(SDPBackend.MATH):
            u,du = torch.func.jvp(lambda a,b:self(a,r,b,features), (z,t), (v,torch.ones_like(t)))
        return F.mse_loss(u,(v-(t-r)[scale]*du).detach())


def heads(shared=96, scale="tiny"):
    if scale == "reference":
        return ActionHead(shared,3072,27,24), VideoFlow(shared,1536,36,12)
    if scale != "tiny":
        raise ValueError(scale)
    return ActionHead(shared,64,2,4), VideoFlow(shared,64,2,4)
