"""Four outward stereo pairs -> measured depth -> occupancy -> orbit view."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

W, H, F = 320, 176, 145.
BASELINE, STEP = .30, .18
LOW = np.array([-12., -12., -.12])
SHAPE = (134, 134, 23)
# World is Z-up. Each box is (lower corner, upper corner, base RGB).
BOXES = [
    ((-2, 6, 0), (1, 8, 1.9), (57, 155, 181)),
    ((4, 1, 0), (6, 4.8, 1.5), (233, 158, 69)),
    ((-6.5, -4, 0), (-5, -.6, 2.4), (102, 154, 120)),
    ((1, -8, 0), (4, -6, 1.1), (201, 103, 88)),
    ((-3.2, 3.8, 0), (-2.8, 4.2, 1), (233, 181, 65)),
]


def rig(angle):
    fwd = np.array([np.sin(angle), np.cos(angle), 0.])
    right = np.array([np.cos(angle), -np.sin(angle), 0.])
    return np.stack([right, [0., 0., -1.], fwd], axis=1)


def photograph(origin, axes):
    y, x = np.indices((H, W))
    rays = np.stack([(x-(W-1)/2)/F, (y-(H-1)/2)/F, np.ones_like(x)], -1) @ axes.T
    with np.errstate(divide="ignore", invalid="ignore"):
        ground = -origin[2] / rays[..., 2]
    depth = np.where(ground > 0, ground, np.inf)
    base = np.full((H, W, 3), [103., 115., 125.])
    for lo, hi, color in BOXES:
        with np.errstate(divide="ignore", invalid="ignore"):
            t1, t2 = (lo-origin)/rays, (hi-origin)/rays
        enter = np.minimum(t1, t2).max(-1)
        leave = np.maximum(t1, t2).min(-1)
        hit = (enter > 0) & (leave >= enter) & (enter < depth)
        depth[hit], base[hit] = enter[hit], color
    valid = depth < 20
    xyz = origin + np.where(valid, depth, 0)[..., None]*rays
    # Deterministic world-space texture gives rectified stereo something to match.
    ix, iy, iz = np.floor(xyz * 13 + .001).astype(int).transpose(2, 0, 1)
    noise = np.mod(np.sin(ix*12.9898 + iy*78.233 + iz*32.721)*43758.5453, 1.)
    base += (noise[..., None]-.5)*50
    stripe = (abs(np.mod(xyz[..., 0]+1.5, 3.)-1.5) < .065) & (xyz[..., 2] < .01)
    base[stripe] += 67
    base[~valid] = (171, 187, 199)
    return base.clip(0, 255).astype("uint8"), depth


def stereo(left, right, focal_baseline=F*BASELINE, max_depth=12., disparities=32):
    def match(a, b, low):
        matcher = cv2.StereoSGBM_create(minDisparity=low, numDisparities=disparities,
                    blockSize=5, P1=200, P2=800, uniquenessRatio=9,
                    speckleWindowSize=40, speckleRange=2)
        return matcher.compute(a, b).astype(float)/16
    a, b = [cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) for im in (left, right)]
    disp, back = match(a, b, 0), match(b, a, -disparities)
    y, x = np.indices(a.shape)
    xr = np.rint(x-disp).astype(int).clip(0, a.shape[1]-1)
    valid = (disp > focal_baseline/max_depth) & (back[y, xr] > -disparities) & (abs(disp + back[y, xr]) < 1.)
    depth = np.full(a.shape, np.nan)
    depth[valid] = focal_baseline/disp[valid]
    return depth


def fuse(images, depths, centers):
    indices = np.indices(SHAPE).reshape(3, -1).T
    xyz = LOW + (indices+.5)*STEP
    votes = np.zeros(len(xyz), int)
    free = np.zeros(len(xyz), bool)
    colors = np.zeros((len(xyz), 3), float)
    for i, (rgb, depth, origin) in enumerate(zip(images, depths, centers)):
        camera = np.einsum("ni,ij->nj", xyz-origin, rig(i*np.pi/2))
        z = camera[:, 2]
        uv = F*camera[:, :2]/np.maximum(z, .01)[:, None] + [(W-1)/2, (H-1)/2]
        u, v = np.rint(uv).astype(int).T
        ids = np.flatnonzero((z > .2) & (u >= 0) & (u < W) & (v >= 0) & (v < H))
        measured = depth[v[ids], u[ids]]
        delta = measured-z[ids]
        hit = ids[abs(delta) <= STEP*.65]
        votes[hit] += 1
        colors[hit] += rgb[v[hit], u[hit]]
        free[ids[delta > STEP*.65]] = True
    mask = votes > 0
    grid = np.full(len(xyz), -1, np.int8)
    grid[free], grid[mask] = 0, 1
    return grid.reshape(SHAPE), xyz[mask], (colors[mask]/votes[mask,None]).astype('uint8')


def orbit(points, colors):
    frames = []
    for angle in np.arange(36)*np.pi/18:
        ax = rig(angle)
        right, fwd = ax[:, 0], ax[:, 2]
        up = np.array([0., 0., 1.])
        basis = np.stack([right, .67*fwd+.74*up], axis=1)
        order = np.argsort(np.einsum("ni,i->n", points, .74*fwd-.67*up))[::-1]
        xy = np.einsum("ni,ij->nj", points, basis) * [12.5, -12.5] + [250, 163]
        frame = Image.new("RGB", (500, 324), "#172027")
        draw = ImageDraw.Draw(frame)
        for j in order:
            x, y = xy[j]
            draw.rectangle((int(x)-1, int(y)-1, int(x)+2, int(y)+2), fill=tuple(colors[j]))
        car = np.array([[-.8,-1.7,0],[.8,-1.7,0],[.8,1.7,0],[-.8,1.7,0]])
        corners = car @ basis * [12.5,-12.5] + [250,163]
        draw.polygon([tuple(p) for p in corners], fill="#b6c6d2", outline="#182028")
        draw.text((12,12), "OPEN SURROUND VIEW  /  STEREO OCCUPANCY", fill="#dae5ec")
        draw.text((12,300), "Synthetic rig    |    18 cm voxels    |    fixed scene", fill="#afc2cf")
        frames.append(frame)
    return frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("run"))
    out = parser.parse_args().out
    out.mkdir(parents=True, exist_ok=True)
    lefts, depths, centers, errors, rows = [], [], [], [], []
    reference_count = 0
    for i in range(4):
        axes = rig(i*np.pi/2)
        center = axes[:, 2]*.75 + [0.,0.,1.35]
        left_origin = center - BASELINE/2*axes[:, 0]
        left, truth = photograph(left_origin, axes)
        right, _ = photograph(center + BASELINE/2*axes[:, 0], axes)
        depth = stereo(left, right)
        supported = (truth >= F*BASELINE/31) & (truth < 12)
        reference_count += int(supported.sum())
        valid = supported & np.isfinite(depth)
        errors.extend(abs(depth[valid]-truth[valid]))
        rows.append(np.concatenate([left, right], axis=1))
        lefts.append(left)
        depths.append(depth)
        centers.append(left_origin)
    grid, points, colors = fuse(lefts, depths, centers)
    np.savez_compressed(out/"occupancy.npz", grid=grid, origin=LOW, voxel_m=STEP)
    with (out/"surface.ply").open("w") as stream:
        stream.write("ply\nformat ascii 1.0\nelement vertex %d\n" % len(points))
        stream.write("property float x\nproperty float y\nproperty float z\n")
        stream.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        np.savetxt(stream, np.c_[points, colors], fmt="%.3f %.3f %.3f %d %d %d")
    Image.fromarray(np.concatenate(rows)).save(out/"stereo.png")
    frames = orbit(points, colors)
    frames[0].save(out/"orbit.gif", save_all=True, append_images=frames[1:], duration=100, loop=0)
    stats = dict(valid_depth_fraction=round(len(errors)/reference_count,3),
                 median_depth_error_m=round(float(np.median(errors)),3),
                 p95_depth_error_m=round(float(np.percentile(errors,95)),3),
                 occupied_voxels=int((grid==1).sum()))
    (out/"metrics.json").write_text(json.dumps(stats, indent=2)+"\n")
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
