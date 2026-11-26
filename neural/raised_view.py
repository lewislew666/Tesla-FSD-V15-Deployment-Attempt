"""A small, fixed-camera MeanFlow experiment on real stereo road footage."""
import argparse
import io
import json
import math
import sys
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.nn import functional as F

from reality import SharedVision
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"baselines"/"stereo"))
from demo import stereo
from kitti import calibration, DRIVE

W, H = 224, 96
TRAIN, TEST = list(range(48)), list(range(56,72))
ANGLE = math.radians(14)
C, S = math.cos(ANGLE), math.sin(ANGLE)
FOCAL, RISE, BACK = 153., 1.5, 1.
CX, CY = (W-1)/2, (H-1)/2


def make_reference(rgb, depth, k):
    """Forward project stereo surfaces. Keep nearest point and an explicit mask."""
    vv, uu = np.nonzero(np.isfinite(depth))
    z = depth[vv, uu]
    x = (uu-k[0,2])*z/k[0,0]
    y = (vv-k[1,2])*z/k[1,1] + RISE
    z = z + BACK
    virtual_z = S*y + C*z
    u = np.rint(FOCAL*x/virtual_z + CX).astype(int)
    v = np.rint(FOCAL*(C*y-S*z)/virtual_z + CY).astype(int)
    inside = (u>=0)&(u<W)&(v>=0)&(v<H)&(virtual_z>0)
    pos = v[inside]*W+u[inside]
    nearest = np.full(W*H,np.inf)
    np.minimum.at(nearest,pos,virtual_z[inside])
    keep = virtual_z[inside] == nearest[pos]
    image = np.zeros((H*W,3),np.uint8)
    image[pos[keep]] = rgb[vv[inside][keep],uu[inside][keep]]
    return image.reshape(H,W,3), np.isfinite(nearest).reshape(H,W)


def projection_grid(k, original_hw):
    """Fixed flat-road coordinate prior (camera height assumed 1.65 m)."""
    h,w = original_hw
    v,u = np.indices((H,W),dtype=np.float32)
    a,b = (u-CX)/FOCAL, (v-CY)/FOCAL
    dy,dz = C*b+S, C-S*b
    distance = np.minimum(90., (1.65+RISE)/np.maximum(dy,.001))
    x,y,z = distance*a, distance*dy-RISE, distance*dz-BACK
    px = k[0,0]*x/z + k[0,2]
    py = k[1,1]*y/z + k[1,2]
    return torch.from_numpy(np.stack([2*(px+.5)/w-1,2*(py+.5)/h-1],-1)).float()


def read_data(args):
    with ZipFile(args.calib) as archive:
        params = calibration(archive,"calib_cam_to_cam.txt")
    left_p,right_p = [params[key].reshape(3,4) for key in ("P_rect_02","P_rect_03")]
    fb = left_p[0,3]-right_p[0,3]
    encoder = SharedVision.from_checkpoint(str(args.checkpoint),meta_repo=args.meta_repo).encoder.model.eval()
    dense, refs, masks, inputs, flat, patches = [],[],[],[],[],[]
    with ZipFile(args.drive) as archive, torch.no_grad():
        for frame in TRAIN+TEST:
            pair = [np.asarray(Image.open(io.BytesIO(archive.read(f"{DRIVE}/{cam}/data/{frame:010}.png"))).convert("RGB"))
                    for cam in ("image_02","image_03")]
            rgb = pair[0]
            depth = stereo(*pair,fb,55.,128)
            ref,mask = make_reference(rgb,depth,left_p)
            grid = projection_grid(left_p,rgb.shape[:2])
            x = torch.from_numpy(np.stack([cv2.resize(im,(448,160),interpolation=cv2.INTER_AREA) for im in pair])).permute(0,3,1,2).float()/255
            if frame in TRAIN:
                patches.append(F.unfold(x[:1],16,stride=16)[0].T)
            x = (x-x.new_tensor([.485,.456,.406])[None,:,None,None])/x.new_tensor([.229,.224,.225])[None,:,None,None]
            low,high = encoder.get_intermediate_layers(x,n=[1,11],reshape=True)
            # Keep the source view separate for a linear texture readout.
            first = encoder.patch_embed(x[:1])[0].permute(2,0,1)
            dense.append(torch.cat((torch.cat([low,high],1).mean(0),first),0))
            refs.append(ref)
            masks.append(mask)
            inputs.append(cv2.resize(rgb,(W,H), interpolation=cv2.INTER_AREA))
            px = ((grid[...,0].numpy()+1)*rgb.shape[1]/2-.5).astype(np.float32)
            py = ((grid[...,1].numpy()+1)*rgb.shape[0]/2-.5).astype(np.float32)
            flat.append(cv2.remap(rgb,px,py,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE))
    def to_images(frames):
        return torch.from_numpy(np.stack(frames)).permute(0,3,1,2).float()/255
    return torch.stack(dense), to_images(refs), torch.from_numpy(np.stack(masks))[:,None].float(), to_images(inputs), to_images(flat), grid, torch.cat(patches)


class RaisedFlow(nn.Module):
    """Conditional RGB velocity; target-plane sampling uses a fixed calibrated view."""
    def __init__(self, texture=True):
        super().__init__()
        self.texture = texture
        self.shared = nn.Sequential(nn.Conv2d(768,96,1),nn.GroupNorm(8,96),nn.GELU())
        self.temporal = nn.TransformerEncoderLayer(96,4,192,dropout=0,batch_first=True)
        self.decode = nn.Sequential(nn.Conv2d(96,96,3,padding=1),nn.GELU(),
                                    nn.Conv2d(96,64,3,padding=1),nn.PixelShuffle(2),nn.GELU(),
                                    nn.Conv2d(16,48,3,padding=1),nn.PixelShuffle(4))
        self.residual = nn.Sequential(nn.Conv2d(8,16,3,padding=1),nn.SiLU(),nn.Conv2d(16,3,3,padding=1))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

        if self.texture:
            self.tex_readout = nn.Linear(384,3*16*16).requires_grad_(False)
            self.adjust = nn.Sequential(nn.Conv2d(96,32,3,padding=1), nn.GELU(),nn.Conv2d(32,3,1))
            nn.init.zeros_(self.adjust[-1].weight)
            with torch.no_grad():
                self.adjust[-1].bias.copy_(torch.tensor([0.,0.,2.]))

    @torch.no_grad()
    def fit_texture(self,features,patches):
        # Fit only on the training cameras. The linear map belongs to this experiment.
        x = features[:,768:].permute(0,2,3,1).reshape(-1,384).double()
        y = patches.double()
        mx,my = x.mean(0),y.mean(0)
        x,y = x-mx,y-my
        beta = torch.linalg.solve(x.T@x + .1*torch.eye(384,dtype=x.dtype),x.T@y)
        self.tex_readout.weight.copy_(beta.T)
        self.tex_readout.bias.copy_(my-mx@beta)

    def condition(self, features, grid):
        x = self.shared(features[:,:768])
        summary = x.mean((2,3))[None]
        causal = torch.ones((len(x),len(x)),dtype=torch.bool,device=x.device).triu(1)
        x = x + self.temporal(summary,src_mask=causal)[0,:,:,None,None]
        g = F.avg_pool2d(grid.permute(2,0,1)[None],8,8).permute(0,2,3,1)
        sample = F.grid_sample(x,g.expand(len(x),-1,-1,-1),padding_mode="border",align_corners=False)
        coarse = self.decode(sample).sigmoid()
        if not self.texture:
            return coarse
        patch_rgb = self.tex_readout(features[:,768:].permute(0,2,3,1)).permute(0,3,1,2)
        source = F.pixel_shuffle(patch_rgb,16)
        adjustments = F.interpolate(self.adjust(sample),size=(H,W),mode="bilinear",align_corners=False)
        shifts = .10*adjustments[:,:2].tanh().permute(0,2,3,1)
        detail = F.grid_sample(source,grid[None]+shifts,padding_mode="border",align_corners=False)
        alpha = adjustments[:,2:3].sigmoid()
        return (alpha*detail+(1-alpha)*coarse).clamp(0,1)

    def velocity(self,z,r,t,cond):
        time = torch.cat((r,t),1).expand(-1,-1,H,W)
        return (z-cond)/(.15+.85*t) + self.residual(torch.cat((z,cond,time),1))

    def generate(self,previous,cond,noise):
        source = previous + .2*noise
        t = torch.ones((len(source),1,1,1))
        return (source-self.velocity(source,0*t,t,cond)).clamp(0,1)

    def objective(self,previous,cond,target,mask):
        source = previous + .2*torch.randn_like(previous)
        r,t = torch.rand(2,1,1,1,1).sort(dim=0).values.expand(-1,len(target),-1,-1,-1).contiguous()
        v = source-target
        z = target+t*v
        u,du = torch.func.jvp(lambda a,b:self.velocity(a,r,b,cond),(z,t),(v,torch.ones_like(t)))
        # Exclude unknown target pixels; sample direct one-step endpoints as an auxiliary loss.
        err = (u-(v-(t-r)*du).detach()).square()
        direct = self.generate(previous,cond,torch.randn_like(target))
        endpoint = (direct-target).abs()
        return ((err + 3*endpoint)*mask).sum()/(3*mask.sum())


def psnr(output,target,mask):
    mse = ((output-target).square()*mask).sum()/(3*mask.sum())
    return -10*math.log10(float(mse))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("drive","calib","checkpoint","meta-repo","out"):
        p.add_argument("--"+k,type=Path,required=True)
    p.add_argument("--coarse-only",action="store_true",help="Disable the DINO patch texture readout")
    p.add_argument("--load-head",type=Path,help="Evaluate an existing trained head without updating any weights")
    p.add_argument("--steps",type=int,default=500)
    args=p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(37)
    dense,refs,masks,inputs,flat,grid,patches = read_data(args)
    model=RaisedFlow(texture=not args.coarse_only)
    if model.texture and not args.load_head:
        model.fit_texture(dense[:len(TRAIN)],patches)
    opt=torch.optim.AdamW(model.parameters(),lr=.0015)
    if args.load_head:
        model.load_state_dict(torch.load(args.load_head,map_location="cpu",weights_only=True))
    for step in range(0 if args.load_head else args.steps):
        i=int(torch.randint(0,len(TRAIN)-7,()))
        cond=model.condition(dense[i:i+8],grid)
        with torch.no_grad():
            if i>=8:
                past=model.condition(dense[i-8:i],grid)
                prev=model.generate(torch.zeros_like(past),past,torch.randn_like(past))
            else:
                prev=torch.zeros_like(cond)
        loss=model.objective(prev,cond,refs[i:i+8],masks[i:i+8])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        opt.step()
        if (step+1)%100==0:
            print(f"step {step+1}: loss {loss.item():.4f}",flush=True)
    args.out.mkdir(parents=True,exist_ok=True)
    if not args.load_head:
        torch.save(model.state_dict(),args.out/"raised-flow.pth")
    torch.manual_seed(1234)
    model.eval()
    pred=[]
    with torch.no_grad():
        prev=torch.zeros_like(refs[:8])
        for start in range(len(TRAIN),len(dense),8):
            cond=model.condition(dense[start:start+8],grid)
            prev=model.generate(prev,cond,torch.randn_like(cond))
            pred.append(prev)
    pred=torch.cat(pred)
    target,mask=refs[len(TRAIN):],masks[len(TRAIN):]
    avg=(refs[:len(TRAIN)]*masks[:len(TRAIN)]).sum(0)/masks[:len(TRAIN)].sum(0).clamp_min(1)
    # The ground-plane prior reaches outside the source camera near the bottom.
    # A display mask from calibration alone avoids showing border padding as road.
    display_mask = grid.abs().amax(-1) <= 1
    shown = torch.where(display_mask[None,None],pred,torch.tensor([.09,.13,.16])[None,:,None,None])
    stats=dict(display_fraction=float(display_mask.float().mean()),train_frames=TRAIN,test_frames=TEST,training_updates_this_run=(0 if args.load_head else args.steps),
               trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
               width=W,height=H,virtual_height_offset_m=RISE,setback_m=BACK,pitch_deg=14,
               valid_pixel_fraction=mask.mean().item(),psnr_learned_db=psnr(pred,target,mask),
               psnr_training_mean_db=psnr(avg,target,mask),psnr_ground_plane_warp_db=psnr(flat[len(TRAIN):],target,mask))
    (args.out/"metrics.json").write_text(json.dumps(stats,indent=2)+"\n")
    def image(x):
        return Image.fromarray((255*x.permute(1,2,0).clamp(0,1)).byte().numpy()).resize((336,144),Image.Resampling.BICUBIC)
    movie=[]
    for n,number in enumerate(TEST):
        board=Image.new("RGB",(1024,191),"#15232d")
        draw=ImageDraw.Draw(board)
        for j,(label,view) in enumerate([("Recorded front camera",inputs[len(TRAIN)+n]),
                           ("Learned view / gray = outside projection",shown[n]),("Stereo-projected reference",target[n])]):
            x=4+j*340
            draw.text((x,6),label,fill="white")
            board.paste(image(view),(x,23))
        draw.text((4,173),f"KITTI 0001 / frame {number:03d}    |    Reference: black = no stereo observation",fill="white")
        movie.append(board)
    movie[0].save(args.out/"preview.png")
    movie[0].save(args.out/"sequence.gif",save_all=True,append_images=movie[1:],duration=100,loop=0)
    print({k:v for k,v in stats.items() if not k.endswith('frames')},flush=True)

if __name__=="__main__":
    main()
