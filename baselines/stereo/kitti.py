"""Reconstruct KITTI Raw front stereo and evaluate on independent Velodyne samples."""
import argparse
import io
import json
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
from PIL import Image, ImageDraw

from demo import stereo

DRIVE = "2011_09_26/2011_09_26_drive_0001_sync"
STEP = 0.20
# Local metric coordinates: right, forward, up, measured from the left optical center.
ORIGIN = np.array([-18., 0., -3.])
SHAPE = (180, 200, 50)


def calibration(archive, filename):
    out = {}
    for line in archive.read("2011_09_26/" + filename).decode().splitlines():
        key, value = line.split(":", 1)
        if key != "calib_time":
            out[key] = np.array([float(v) for v in value.split()])
    return out


def compare_lidar(scan, transform, projection, depth, grid):
    points = np.frombuffer(scan, "<f4").reshape(-1, 4)[:, :3]
    rect = np.einsum("nj,ij->ni", points, transform[:, :3]) + transform[:, 3]
    projected = np.einsum("nj,ij->ni", rect, projection[:, :3]) + projection[:, 3]
    projected = projected[projected[:, 2] > .1]
    z = projected[:, 2]
    u, v = np.rint(projected[:, :2] / z[:, None]).astype(int).T
    h, w = depth.shape
    keep = (u >= 0) & (v >= 0) & (u < w) & (v < h)
    reference = np.full((h, w), np.inf)
    np.minimum.at(reference, (v[keep], u[keep]), z[keep])
    valid = (reference >= 3.) & (reference <= 40.)
    paired = valid & np.isfinite(depth)
    # Evaluate one reference return per image pixel, at the pixel-center ray.
    yy, xx = np.nonzero(valid)
    d = reference[yy, xx]
    ref_xyz = np.column_stack(((xx-projection[0,2])*d/projection[0,0], d,
                                -(yy-projection[1,2])*d/projection[1,1]))
    ijk = np.floor((ref_xyz-ORIGIN)/STEP).astype(int)
    inside = np.all((ijk >= 0) & (ijk < SHAPE), axis=1)
    labels = grid[tuple(ijk[inside].T)]
    counts = {s:int(np.sum(labels==i)) for s,i in (("unknown",-1),("free",0),("occupied",1))}
    return int(valid.sum()), abs(depth[paired]-reference[paired]), counts


def voxel_projection(intrinsics, shape):
    """Precompute projection of voxel centers into the left rectified image."""
    h, w = shape
    xyz = ORIGIN + (np.indices(SHAPE, dtype=np.float32).reshape(3, -1).T+.5)*STEP
    x, z, up = xyz.T
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    u = np.rint(fx*x/z + intrinsics[0, 2]).astype(int)
    v = np.rint(-fy*up/z + intrinsics[1, 2]).astype(int)
    ids = np.flatnonzero((u >= 0) & (u < w) & (v >= 0) & (v < h))
    return ids, u[ids], v[ids], xyz


def occupancy(depth, rgb, lookup):
    ids, u, v, xyz = lookup
    delta = depth[v, u] - xyz[ids, 1]
    grid = np.full(len(xyz), -1, 'int8')
    grid[ids[delta > STEP*.65]] = 0
    surface = abs(delta) <= STEP*.65
    grid[ids[surface]] = 1
    return grid.reshape(SHAPE), xyz[ids[surface]], rgb[v[surface], u[surface]]


def panel(rgb, points, colors, frame_id):
    canvas = Image.new("RGB", (820, 550), "#14212b")
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 10), f"KITTI RAW    /    FRONT LEFT CAMERA      |      FRAME {frame_id:03d}", fill="white")
    canvas.paste(Image.fromarray(rgb).resize((820, 247), Image.Resampling.LANCZOS), (0, 32))
    draw.text((16, 289), "IMAGE-DERIVED OCCUPIED SURFACES    /    ELEVATED VIEW", fill="white")
    # Orthographic virtual view rotated sideways and tilted down.
    x, y, z = points.T
    u = np.rint(369 + 12*(.95*x+.31*y)).astype(int)
    v = np.rint(535 - 12*(.50*(.95*y-.31*x)+.866*z)).astype(int)
    region = np.array(canvas)
    order = np.argsort(.866*(.95*y-.31*x)-.5*z)[::-1]
    # Paint far to near; repeated pixels use the nearest observed surface.
    for dx, dy in [(0,0),(1,0),(0,1),(1,1)]:
        xx, yy = u[order]+dx, v[order]+dy
        keep = (xx>=0)&(xx<820)&(yy>=313)&(yy<550)
        region[yy[keep], xx[keep]] = colors[order[keep]]
    return Image.fromarray(region)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive", type=Path, required=True)
    parser.add_argument("--calib", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("run-kitti"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with ZipFile(args.calib) as archive:
        cameras = calibration(archive, "calib_cam_to_cam.txt")
        laser = calibration(archive, "calib_velo_to_cam.txt")
    p2, p3 = (cameras[k].reshape(3,4) for k in ("P_rect_02","P_rect_03"))
    fb = p2[0,3] - p3[0,3]
    transform = cameras["R_rect_00"].reshape(3,3) @ np.column_stack((laser["R"].reshape(3,3), laser["T"]))
    errors, panels, rows = [], [], []
    lookup = None
    with ZipFile(args.drive) as archive:
        for frame in range(0,108,6):
            name = f"{frame:010d}"
            images = []
            for side in ["image_02","image_03"]:
                raw = archive.read(f"{DRIVE}/{side}/data/{name}.png")
                images.append(np.asarray(Image.open(io.BytesIO(raw)).convert("RGB")))
            left, right = images
            depth = stereo(left, right, fb, 40., 128)
            if lookup is None:
                lookup = voxel_projection(p2[:,:3], depth.shape)
            grid, points, colors = occupancy(depth, left, lookup)
            panels.append(panel(left, points, colors, frame))
            if frame == 0:
                np.savez_compressed(args.out/"frame000.npz", grid=grid, origin=ORIGIN, voxel_m=STEP,
                                    points=points, rgb=colors)
                with (args.out/"frame000.ply").open("w") as f:
                    f.write(f"ply\nformat ascii 1.0\nelement vertex {len(points)}\n")
                    f.write("property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
                    np.savetxt(f, np.c_[points, colors], fmt="%.2f %.2f %.2f %d %d %d")
                Image.fromarray(np.concatenate(images,axis=0)).save(args.out/"stereo000.png")
                panels[0].save(args.out/"preview.jpg",quality=93)
            raw_scan = archive.read(f"{DRIVE}/velodyne_points/data/{name}.bin")
            count, err, hits = compare_lidar(raw_scan, transform, p2, depth, grid)
            rows.append(dict(frame=frame, lidar_pixels=count, matched=len(err), lidar_endpoint_states=hits,
                             median_m=float(np.median(err)), occupied=int((grid==1).sum())))
            errors.append(err)
    panels[0].save(args.out/"sequence.gif", save_all=True, append_images=panels[1:], duration=600, loop=0)
    err = np.concatenate(errors)
    count = sum(r["lidar_pixels"] for r in rows)
    endpoint_states = {k:sum(r["lidar_endpoint_states"][k] for r in rows) for k in ("unknown","free","occupied")}
    stats = dict(lidar_endpoint_states=endpoint_states, drive=DRIVE, depth_range_m=[3,40], lidar_pixels=count, matched=len(err),
                 coverage=len(err)/count, median_m=float(np.median(err)),
                 p95_m=float(np.percentile(err,95)), frames=rows)
    (args.out/"metrics.json").write_text(json.dumps(stats,indent=2)+"\n")
    print({k:v for k,v in stats.items() if k!="frames"})


if __name__ == "__main__":
    main()
