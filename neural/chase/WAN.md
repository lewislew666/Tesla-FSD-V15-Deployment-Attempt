# Wan VACE masked repaint on KITTI

This optional experiment uses the RGB geometry guide as Wan's video input. The white pixels in the per-frame mask mark gaps in stereo coverage, with the procedural car excluded. The final composite feathers changes inside those gaps and copies all remaining pixels from the original renderer.

[Synchronized comparison](wan-result/comparison.mp4) · [Wan composite](wan-result/wan-chase.mp4) · [Roadside crops](wan-result/roadside-comparison.jpg) · [Measured results](wan-result/metrics.json)

On an Apple M5 Max with 36 GB of unified memory, the 448 × 256 BF16 trial took 121.22 seconds for generation and 137.40 seconds for the full Wan invocation, excluding weight downloads and the subsequent comparison export. Peak MLX allocation was 8.23 GiB. The 17 KITTI exposures span 1.754 seconds of playback. The editable area averaged 12.94% of the full frame. The flow-warped RGB difference on overlapping edited regions averaged 14.07 for the original and 12.54 for the composite (0–255 scale). Smoothing can lower this metric. In the paired crops, texture changes are subtle and the stretched roadside geometry persists.

The Model Y preview reuses this same Wan run with a new procedural vehicle composite. [Model Y clip](wan-result/wan-model-y.mp4) · [Side-by-side vehicle comparison](wan-result/vehicle-comparison.mp4) · [Still frame](wan-result/vehicle-comparison.jpg)

## Reproduce the trial

The published 17-frame trial used `--vehicle sedan`. The comparator reads the vehicle selection from the render metrics and treats earlier caches as sedan runs.

Start with the geometry cache, uncompressed frames and timing CSV from a complete run of `neural/chase/render.py`. Run the preparation and comparison commands in the existing renderer environment. For the MLX step on Apple Silicon, install `mlx-gen==0.40.1` in a separate Python 3.12 environment and put its executable and FFmpeg on your path. Get the Diffusers checkpoint from the [Wan team](https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B-diffusers). The local checkpoint directory name should contain `Wan2.1-VACE-1.3B`.

From the repository root:

```sh
python neural/chase/wan_compare.py prepare \
  --cache cache-chase --timing run-chase/timing.csv --out wan-trial

mlxgen-generate-wan \
  --model /path/to/Wan2.1-VACE-1.3B-diffusers \
  --video-path wan-trial/guide.mp4 --video-mask-path wan-trial/mask.mp4 \
  --vace-masked-region repaint --solver unipc --low-ram \
  --width 448 --height 256 --frames 17 --fps 10 --steps 16 --guidance 4 --seed 722 \
  --prompt 'Photorealistic daytime footage from a fixed elevated chase camera moving forward along the same German suburban street. Match the source road, lane markings, fences, parked cars, trees, camera motion, colors and shadows. Restore coherent natural textures in the masked road and roadside surfaces.' \
  --negative-prompt 'new traffic, extra vehicles, extra lane markings, camera cut, change of viewpoint, flicker, smeared pavement, distorted fence' \
  --metadata --output wan-trial/wan.mp4

python neural/chase/wan_compare.py compare \
  --cache cache-chase --timing run-chase/timing.csv --out wan-trial --raw wan-trial/wan.mp4
```

The default interval is frames 064–080. The two conditioning streams use an intermediate 10 fps clock so the model sees one image and one mask per index. The comparator restores the original KITTI presentation timestamps and checks the final packet duration. It saves the side-by-side clip, the standalone composite and per-run metrics. The temporal metric measures mean RGB disagreement after reprojecting consecutive frames over overlapping editable pixels. The reprojection is approximate in occluded regions, and there is no independent overhead reference for this drive.
