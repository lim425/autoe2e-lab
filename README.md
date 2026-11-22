# autoe2e-lab

Run the open-source [AutoE2E](https://github.com/autowarefoundation/auto_e2e) end-to-end
driving model on recorded driving data, and visualise what it predicts next to what the
human driver actually did.

AutoE2E takes six surround cameras (plus 3.5 s of camera history), a map raster, the
planned route and the car's recent motion, and outputs the next 6.4 s of driving as
64 steps of acceleration and curvature. This repo loads the released checkpoint, feeds it
scenes from the KITScenes validation set, and draws the result:

![Uploading with_route_vs_human_10hz.gif…]()


- **purple** = the human driver's real path (ground truth)
- **green** = the AutoE2E prediction

on the front camera image and on a top-down map, frame by frame, as images and video.

Offline replay of recorded data only. Nothing here controls a vehicle.

## Status

| Dataset | Checkpoint | Status |
|---|---|---|
| KITScenes-Multimodal validation (AutoE2E packed v3.5) | `kitscenes-epoch-5` | ✅ working |

## Findings

All numbers are **open-loop**: the car always follows the recorded human, and at each
moment the model predicts the next seconds. "Error" is ADE @ 3 s, the average distance
between the predicted and the real path over the next 3 s. Few drives were tested, so
treat these as observations, not statistics. Raw numbers and plots: [`results/`](results/).

**1. Accuracy is in line with the authors.** Mean ADE @ 3 s on three drives:
0.78 m (2 s, 20 moments), 0.85 m (31 s, 62 moments), 0.79 m (51 s, 102 moments). The
authors report 0.744 m on 202 keyframes of the full validation set. The model is very good
on single-road stretches (0.10–0.20 m) and worst at forks and roundabouts (up to ~2 m).

**2. Route data is missing for 17 of 140 validation drives**, including the 5 longest
(`scripts/scan_routes.py`). On those drives the model has to guess at junctions and
sometimes takes the wrong branch or roundabout exit.

**3. Route on/off.** Same moments, with the dataset's route and with the route switched off
through the model's own validity gate. The authors' paper notes this paired comparison is
not yet published.

| Drive | Moments | With route | Without route | Route helped / hurt |
|---|---|---|---|---|
| `part-d646a7f28b51e27f` (34 s, 10 Hz) | 340 | **0.69 m** | 0.76 m | 46% / 31% |
| `part-8028dd4a50d07877` (31 s, 2 Hz) | 62 | 0.78 m | **0.74 m** | 53% / 29% |

The route clearly changes the output and helps most at turns where there is a choice
(up to 1.2 m better), but the average gain is small and not consistent between drives. In
both drives the route sometimes made the model turn too sharply in left turns (worst
case 2.35 m with route vs 0.37 m without).

**4. Cameras matter less than expected** (`scripts/06_sanity_checks.py`, 3 moments):
normal 0.56 m, identical repeat run, random weights 4.40 m, all cameras black 0.72 m,
cameras shuffled 0.66 m. Blacking out the cameras changes the model's internal camera BEV
by ~70% but the trajectory only a little, so the planner leans on the map, the route and
the ego-motion history. A known limit of open-loop evaluation: low error does not prove the
model reacts to what only cameras can see.

**5. The BEV segmentation head is untrained in the released checkpoint** (training loss
0.0, normalisation layers at their initial values), so its output (~0.5 everywhere) cannot
be used to inspect what the model sees.

**6. Speed: ~24 s per moment (~0.04 FPS)** on an RTX 5060 Laptop in FP32, ~2.3 GB GPU
memory. The FPS tables in the AutoE2E repo (19–76 FPS) are for lighter SwinV2/ConvNeXt
configurations without the BEVFormer T8 history, not for the released checkpoint.

## Layout

```
autoe2e_lab/        reusable code
  upstream.py       finds the AutoE2E clone and checks it is at the right commit
  config.py         loads configs/*.yaml (everything dataset-specific)
  data.py           reads scenes in the AutoE2E packed (WebDataset) format
  model.py          loads the checkpoint; predict() mirrors the authors' evaluation loop
  render.py         camera projection and drawing
configs/            one YAML file per dataset
scripts/            numbered experiments (run from the repo root)
outputs/            generated pictures and videos (not committed)
```

## Setup

Tested on Ubuntu 24.04, Python 3.12, NVIDIA RTX 5060 Laptop (8 GB), driver 580.

**1. Folder layout.** This repo expects two siblings:

```
<parent>/
├── autoe2e-lab/     this repo (any folder name)
├── auto_e2e/        AutoE2E source, at the checkpoint's commit
└── autoe2e_data/    downloads
```

**2. AutoE2E source at the exact commit the checkpoints were trained with.** That commit
is no longer on any branch, so fetch it by ID:

```bash
git clone https://github.com/autowarefoundation/auto_e2e.git
cd auto_e2e
git fetch origin 40a75cf8a26b60f6b094125e3924fe0bdb427b02
git switch -c v1-checkpoint 40a75cf8a26b60f6b094125e3924fe0bdb427b02
```

`autoe2e_lab/upstream.py` refuses to run against any other commit.

**3. Python environment.** RTX 50xx GPUs (Blackwell, `sm_120`) need the CUDA 12.8 PyTorch
build. The `cu118` build in the AutoE2E README has no code for these GPUs.

```bash
cd autoe2e-lab
python3 -m venv .venv && source .venv/bin/activate
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install -c constraints.txt -r requirements.txt   # the lock stops pip from replacing torch
pip install -e . --no-deps
```

If your shell sources ROS 2, run `unset PYTHONPATH` before activating, otherwise ROS's
Python packages leak into the venv.

**4. Download the checkpoint and one scene** (~380 MB) from
[Hugging Face](https://huggingface.co/AutowareFoundation/auto_e2e):

```bash
mkdir -p ../autoe2e_data/hf && cd ../autoe2e_data/hf
hf download AutowareFoundation/auto_e2e config.json SHA256SUMS \
    models/kitscenes-epoch-5/checkpoint.pt --local-dir .
grep "models/kitscenes-epoch-5/checkpoint.pt" SHA256SUMS | sha256sum -c -
hf download AutowareFoundation/auto_e2e \
    --include "datasets/kitscenes-val/v3.5/shards/manifest.json" \
    --include "datasets/kitscenes-val/v3.5/shards/part-0b06571c5556d39d-*" \
    --include "datasets/kitscenes-val/v3.5/pool/part-0b06571c5556d39d/*" \
    --local-dir .
```

More scenes: `python scripts/download_scene.py --list`, then `python scripts/download_scene.py <part>`.

## Run

```bash
python scripts/01_load_checkpoint.py      # weights load with an exact layer match
python scripts/02_inspect_sample.py       # look at the inputs; no model involved
python scripts/03_infer_one_sample.py     # one prediction: error + camera/map picture
python scripts/04_make_video.py --part <part> --step 5     # video at 2 Hz (every 5th moment)
python scripts/04_make_video.py --part <part> --step 1     # video at 10 Hz (every moment)
python scripts/05_route_experiment.py --part <part> --step 5   # with vs without route
python scripts/06_sanity_checks.py        # repeat run, random weights, black/shuffled cameras
python scripts/07_look_inside.py          # the model's own camera top-down picture
python scripts/scan_routes.py             # which scenes have a route (reads ~2 MB per scene)
```

Results are written to `outputs/<config>/`. Videos, CSVs and plots carry the rate in their
name (`..._2hz.mp4`, `..._10hz.mp4`), so runs at different rates never overwrite each other.
Predictions are cached per moment and shared between rates: a 10 Hz run reuses every
moment a 2 Hz run already computed.

Long runs (hours) are best started detached from the terminal:

```bash
mkdir -p outputs/logs
nohup python scripts/04_make_video.py --part <part> --step 1 > outputs/logs/<name>.log 2>&1 &
tail -f outputs/logs/<name>.log
```

**Speed:** about 24 s per moment on an RTX 5060 Laptop, using ~2.3 GB of GPU memory.
Almost all of that time is the BEVFormer camera-to-BEV encoder, which runs 8 times per
moment (now plus 7 history frames) and uses plain-PyTorch deformable attention. This is
expected for this architecture: the authors report an 8.2 s forward pass on an L40S
(measured in training mode).

## Credits and licences

- **Code in this repo:** [MIT](LICENSE). It imports AutoE2E (Apache-2.0) and does not copy it.
- **AutoE2E checkpoints:** CC BY-NC-SA 4.0, non-commercial (see the
  [model card](https://huggingface.co/AutowareFoundation/auto_e2e#license)).
- **KITScenes-Multimodal data** (images in any generated output): CC BY-NC 4.0, created by
  the Karlsruhe Institute of Technology, non-commercial use with attribution
  ([kitscenes.com](https://kitscenes.com/)). Generated pictures and videos are not committed.

Not affiliated with the Autoware Foundation or KIT.
