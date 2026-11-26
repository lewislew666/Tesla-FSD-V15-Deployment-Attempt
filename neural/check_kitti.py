"""Verify shared conditioning and a MeanFlow training step using actual KITTI frames."""
import argparse
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
from PIL import Image
from transformers import DINOv3ViTConfig, DINOv3ViTModel
from reality import SharedVision, heads


def load(archive, camera, frames, size):
    path = "2011_09_26/2011_09_26_drive_0001_sync/"
    images = []
    for i in frames:
        buf = archive.read(f"{path}{camera}/data/{i:010d}.png")
        image = Image.open(io.BytesIO(buf)).convert("RGB").resize(size,Image.Resampling.BILINEAR)
        images.append(np.array(image))
    return torch.from_numpy(np.stack(images)).permute(0,3,1,2).float()/255


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--drive",type=Path,required=True)
    p.add_argument("--meta-repo",type=Path,help="Local checkout of facebookresearch/dinov3 for original .pth weights")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint",help="Official Meta .pth, local HF model folder, or HF model ID")
    group.add_argument("--check-wiring",action="store_true",help="Use a tiny random backbone to test shapes and gradients")
    args = p.parse_args()
    torch.manual_seed(7)
    torch.set_num_threads(2)
    if args.check_wiring:
        config = DINOv3ViTConfig(hidden_size=48,intermediate_size=96,num_hidden_layers=1,num_attention_heads=4)
        vision = SharedVision(DINOv3ViTModel(config))
    else:
        vision = SharedVision.from_checkpoint(args.checkpoint,meta_repo=args.meta_repo)
    with ZipFile(args.drive) as z:
        observations = torch.stack([load(z,cam,[0,6],(160,64)) for cam in ("image_02","image_03")],1)[None]
        # Actual right-camera targets, solely to exercise the loss; no elevated-view labels.
        target = load(z,"image_03",range(12,60,6),(64,32))[None]
    features = vision(observations)
    action, flow = heads()
    previous = torch.zeros_like(target)  # cold-start assumption
    a = action(features)
    generated = flow.generate(previous,features,torch.randn_like(previous))
    following = flow.generate(generated.detach(),features,torch.randn_like(previous))
    loss = flow.flow_loss(generated.detach(),target,features)
    loss.backward()
    assert all(torch.isfinite(x).all() for x in [a,following,loss,vision.project.weight.grad])
    assert all(param.grad is None for param in vision.encoder.parameters())
    with torch.device("meta"):
        reference = heads(scale="reference")
    sizes = [sum(p.numel() for p in head.parameters()) for head in reference]
    print(json.dumps(dict(backbone=Path(args.checkpoint).name if args.checkpoint else "RANDOM wiring test only", shared=list(features.shape),
                          actions=list(a.shape), video=list(following.shape), finite_gradients=True,
                          reference_head_parameters=sizes, backbone_pretrained=bool(args.checkpoint), trained_task_heads=False),indent=2))


if __name__=="__main__":
    main()
