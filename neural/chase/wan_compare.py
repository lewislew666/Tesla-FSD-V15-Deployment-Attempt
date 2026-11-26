"""Prepare a masked Wan VACE camera-view trial, then compare it at KITTI timestamps."""
import argparse
import csv
import json
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw

from render import encode, font
from vehicle import vehicle


def read_rows(timing, start, count):
    with timing.open() as f:
        rows = list(csv.DictReader(f))
    offset = next((i for i, r in enumerate(rows) if int(r["source_frame"]) == start), None)
    if offset is None:
        raise ValueError(f"Frame {start} is outside the rendered timing file.")
    selected = rows[offset:offset + count]
    if count < 5 or len(selected) != count or count % 4 != 1:
        raise ValueError("Select an available range containing 4k+1 frames.")
    return offset, selected


def edit_mask(measured, car):
    # Only let Wan fill contiguous holes. Preserve isolated fine stereo returns.
    holes = (~measured & ~car).astype(np.uint8)
    distance = cv2.distanceTransform(holes, cv2.DIST_L2, 5)
    return np.clip((distance - 1.0) / 5.0, 0, 1)


def write_controls(args, rows, car):
    tmp = args.out / "inputs"
    tmp.mkdir(exist_ok=True)
    for k, r in enumerate(rows):
        with np.load(args.cache / f"{int(r['source_frame']):03d}.npz") as frame:
            Image.fromarray(frame["guide"]).save(tmp / f"guide-{k:03d}.png")
            alpha = edit_mask(frame["measured"], car)
            Image.fromarray(np.uint8(alpha > 0) * 255).save(tmp / f"mask-{k:03d}.png")
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    for tag in ("guide", "mask"):
        subprocess.run([ff, "-loglevel", "error", "-y", "-framerate", "10",
                        "-i", str(tmp / f"{tag}-%03d.png"), "-c:v", "libx264",
                        "-crf", "12", "-pix_fmt", "yuv420p", str(args.out / f"{tag}.mp4")], check=True)


def comparison(args, offset, rows, car):
    frames = args.out / "comparison-frames"
    frames.mkdir(exist_ok=True)
    capture = cv2.VideoCapture(str(args.raw))
    if not capture.isOpened():
        raise ValueError(f"Could not read Wan output at {args.raw}")
    measures = []
    previous = None
    residuals = np.zeros(2, dtype=np.float64)
    overlap_pixels = 0
    for k, r in enumerate(rows):
        ok, raw = capture.read()
        if not ok:
            raise ValueError(f"Wan output ended before frame {k}")
        rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        source_id = int(r["source_frame"])
        with np.load(args.cache / f"{source_id:03d}.npz") as frame:
            alpha = edit_mask(frame["measured"], car)
            measured = frame["measured"]
            flow = frame["flowmap"].astype(np.float32)
        original = np.array(Image.open(args.cache / "frames" / f"chase-{offset + k:03d}.png").convert("RGB"))
        # Condition video omits the virtual car. Copy original car pixels after generation.
        wan = cv2.resize(rgb, original.shape[1::-1], interpolation=cv2.INTER_CUBIC)
        a = alpha[:, :, None]
        result = np.rint((1 - a) * original + a * wan).astype(np.uint8)
        changed = alpha > 0
        assert np.array_equal(result[~changed], original[~changed]) and not np.any(changed & (measured | car))
        delta = np.abs(result.astype(np.int16) - original)
        measures.append((changed.mean(), delta[changed].mean() if changed.any() else 0))
        Image.fromarray(result).save(frames / f"wan-chase-{k:03d}.png")
        if previous is not None:
            prev_base, prev_wan, prev_mask = previous
            warped_mask = cv2.remap(prev_mask.astype(np.uint8), flow, None, cv2.INTER_NEAREST)
            h, w = alpha.shape
            inside = ((flow[..., 0] >= 0) & (flow[..., 0] < w - 1) &
                      (flow[..., 1] >= 0) & (flow[..., 1] < h - 1))
            overlap = changed & (warped_mask > 0) & inside
            overlap_pixels += int(overlap.sum())
            for j, (old, new) in enumerate(((prev_base, original), (prev_wan, result))):
                warped = cv2.remap(old, flow, None, cv2.INTER_LINEAR)
                residuals[j] += np.abs(new[overlap].astype(np.int16) - warped[overlap]).sum()
        previous = (original, result, changed)
        canvas = Image.new("RGB", (1312, 434), "#13242a")
        draw = ImageDraw.Draw(canvas)
        for j, (im, label) in enumerate(((original, "Original renderer"), (result, "Wan VACE 1.3B  /  masked repaint"))):
            x = 12 + j * 648
            draw.text((x, 8), label, font=font(21), fill="white")
            canvas.paste(Image.fromarray(im).resize((640, 360), Image.Resampling.LANCZOS), (x, 41))
        draw.text((12, 405), f"KITTI 0001   /   frame {source_id:03d}   /   editable area {100*changed.mean():.1f}%",
                  font=font(16), fill="#b7d5d2")
        canvas.save(frames / f"comparison-{k:03d}.png")
    if capture.read()[0]:
        raise ValueError("Wan produced more frames than the selected KITTI interval")
    capture.release()
    pts = np.array([int(r["mp4_pts_us"]) for r in rows])
    pts -= pts[0]
    durations = np.array([int(r["duration_us"]) for r in rows])
    encode(frames, "comparison", pts, durations, args.out / "comparison.mp4")
    encode(frames, "wan-chase", pts, durations, args.out / "wan-chase.mp4")
    temporal = (residuals / (3 * overlap_pixels)).tolist() if overlap_pixels else [None, None]
    stats = {"frames": len(rows), "first": int(rows[0]["source_frame"]), "last": int(rows[-1]["source_frame"]),
             "duration_s": (int(pts[-1]) + int(durations[-1])) / 1e6,
             "mean_editable_fraction": float(np.mean(measures, axis=0)[0]),
             "mean_rgb_change_in_editable_area_0_to_255": float(np.mean(measures, axis=0)[1]),
             "pre_encoding_max_change_outside_editable_area": 0, "verified_all_presentation_timestamps": True,
             "temporal_overlap_pixel_pairs": overlap_pixels,
             "flow_warped_rgb_mae_baseline": temporal[0], "flow_warped_rgb_mae_wan_composite": temporal[1]}
    (args.out / "comparison.json").write_text(json.dumps(stats, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["prepare", "compare"])
    parser.add_argument("--cache", type=Path, required=True, help="Cache from the original chase render")
    parser.add_argument("--timing", type=Path, required=True, help="timing.csv from that same render")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start", type=int, default=64)
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--raw", type=Path, help="Native generated MP4 for comparison")
    args = parser.parse_args()
    offset, rows = read_rows(args.timing, args.start, args.frames)
    args.out.mkdir(exist_ok=True, parents=True)
    metrics_path = args.timing.with_name("metrics.json")
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    car = vehicle(metrics.get("vehicle", "sedan"))[..., 3] > 0
    if args.operation == "prepare":
        write_controls(args, rows, car)
    elif args.raw is None:
        parser.error("compare requires --raw")
    else:
        comparison(args, offset, rows, car)
