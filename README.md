# Tesla FSD V15 Reality View — An Independent Recreation Attempt with ChaseView

Tesla FSD V15 Reality View inspired ChaseView, an independent attempt to recreate its third-person perspective from real driving footage using stereo vision and diffusion. ChaseView turns real KITTI stereo footage into a synchronized chase-camera video using vehicle motion, DINOv3 features and conditional diffusion.

![A virtual chase camera follows a car along a real KITTI street. The inset shows the front camera at the same moment.](media/chaseview.webp)

[Watch the full clip](neural/chase/result/chase.mp4) · [Compare against the synchronized front camera](neural/chase/result/sync.mp4) · [中文说明](README.zh-CN.md)

## Where the car has just been

When a lane marking disappears beneath the front camera, a chase camera should still see it behind the car. ChaseView uses vehicle poses to place earlier stereo observations into the current view. That supplies actual road texture under and around the illustrated car. A five-step diffusion renderer fills gaps using the projected scene, visual features and the previous output warped into the new camera pose.

The demonstration follows **68 frames over 7.016 seconds** from KITTI Raw drive 0001. The front-camera exposure times determine the presentation timestamps in both MP4s. The [coverage overlay](neural/chase/result/sync.mp4) shows which pixels have geometric support. On average, **57.9% of background pixels** have current or historical stereo support; the rest use approximate projection and completion. The silver sedan is a procedural model.

## Try it

**Watch without installing anything.** The full videos, trained chase renderer and per-frame timing data are in [the demo folder](neural/chase/result/).

**Reproduce the clip.** Use Python 3.12 and run the commands below from this repository. Obtain the original Meta checkpoint named `dinov3_vits16_pretrain_lvd1689m-08c60483.pth` from [DINOv3](https://github.com/facebookresearch/dinov3). Both KITTI archives can stay zipped.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r neural/chase/requirements.txt
git clone https://github.com/facebookresearch/dinov3.git vendor/dinov3
git -C vendor/dinov3 checkout 6876159a11b4df116f30f667f8c9888617df0751

mkdir -p data
base=https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data
curl -fL "$base/2011_09_26_drive_0001/2011_09_26_drive_0001_sync.zip" -o data/drive.zip
curl -fL "$base/2011_09_26_calib.zip" -o data/calib.zip

python neural/chase/render.py \
  --drive data/drive.zip --calib data/calib.zip \
  --meta-repo vendor/dinov3 --checkpoint /path/to/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
  --cache cache-chase --out run-chase
```

Open `run-chase/sync.mp4` to inspect the rendering alongside the recorded input. Add `--train` to fit the included renderer again. Training uses frames 0–35; the published clip uses frames 40–107 of the same drive.

## What the demo measures

| Property | Result |
| --- | --- |
| Virtual camera | 9.2 m behind and 4.7 m above the left camera, pitched down 27° |
| Video | 768 × 432, variable frame timing from KITTI |
| Synchronization | All 68 encoded timestamps matched the expected microsecond timestamps in both MP4s |
| Inference on the development Mac | About 177 ms/frame for diffusion and composite preparation alone; geometry and encoding add latency |
| Scene fidelity | Stereo coverage is measured. There is no recorded overhead target for this clip. Occluded regions can smear or change between frames. |

[Renderer](neural/chase/render.py) · [Two-camera scene reconstruction](neural/chase/scene.py) · [Conditional diffusion](neural/chase/diffusion.py) · [Timing audit](neural/chase/result/metrics.json)

## Credits and reuse

This independent research prototype was inspired by Tesla's Reality View presentation. It provides offline visualization of recorded footage. The repository includes the project-trained diffusion weights. Meta distributes the DINOv3 backbone separately under its own terms. [Tesla demonstration](https://www.youtube.com/watch?v=eNBqkc7V1nU) · [DINOv3](https://github.com/facebookresearch/dinov3)

KITTI Raw was created by Andreas Geiger, Philip Lenz, Christoph Stiller and Raquel Urtasun, *Vision meets Robotics: The KITTI Dataset* (IJRR, 2013). The KITTI footage and KITTI-derived demo imagery, videos, and geometry in this repository are distributed under CC BY-NC-SA 3.0. Modifications include stereo reconstruction, viewpoint reprojection, learned completion and video compositing. [KITTI Raw](https://www.cvlibs.net/datasets/kitti/raw_data.php) · [Data terms](https://www.cvlibs.net/datasets/kitti/) · [CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/)
