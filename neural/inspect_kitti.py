"""Inspect frozen DINOv3 patch features with a shared RGB PCA projection."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
from PIL import Image, ImageDraw
from check_kitti import load
from reality import SharedVision


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--drive", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--meta-repo", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    model = SharedVision.from_checkpoint(str(args.checkpoint), meta_repo=args.meta_repo).eval()
    frame_ids = list(range(0, 108, 6))
    maps = []
    pictures = []
    with ZipFile(args.drive) as archive, torch.inference_mode():
        for frame in frame_ids:
            pair = torch.cat([load(archive, camera, [frame], (640,192))
                              for camera in ("image_02","image_03")])
            grid = model.patch_grid(pair).permute(0,2,3,1).contiguous()
            maps.append(grid)
            pictures.append(pair[0].permute(1,2,0).mul(255).byte().numpy())
    data = torch.stack(maps)
    if not torch.isfinite(data).all() or data.std() == 0:
        raise ValueError("Invalid backbone output")
    flat = data.flatten(0,-2)
    centered = flat-flat.mean(0)
    _, axes = torch.linalg.eigh(centered.T @ centered)
    rgb = centered @ axes[:, -3:].flip(1)
    lo, hi = torch.quantile(rgb, torch.tensor([.02,.98]),dim=0)
    colors = ((rgb-lo)/(hi-lo).clamp_min(1e-6)).clamp(0,1)
    colors = (colors.reshape(*data.shape[:-1],3)*255).byte().numpy()
    args.out.mkdir(parents=True,exist_ok=True)
    # Two source frames show changes over time, with the same projection for all panels.
    board = Image.new("RGB", (960,377), "#19242b")
    draw = ImageDraw.Draw(board)
    for row, idx in enumerate((0, 12)):
        y = 7 + 185*row
        draw.text((8,y), f"KITTI left camera   /   frame {frame_ids[idx]:03d}", fill="white")
        draw.text((488,y),"DINOv3 ViT-S/16   /   patch features (PCA)", fill="white")
        board.paste(Image.fromarray(pictures[idx]).resize((480,144),Image.Resampling.LANCZOS),(0,y+24))
        board.paste(Image.fromarray(colors[idx,0]).resize((480,144),Image.Resampling.NEAREST),(480,y+24))
    board.save(args.out/"features.jpg",quality=95)
    meta = dict(checkpoint=args.checkpoint.name, sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                frames=frame_ids, cameras=["image_02","image_03"], input_hw=[192,640],
                features_shape=list(data.shape), all_finite=True, visualization="joint PCA; 2nd/98th percentile clipping")
    (args.out/"features.json").write_text(json.dumps(meta,indent=2)+"\n")
    print(meta)


if __name__=="__main__":
    main()
