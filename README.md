# autoe2e-lab

Run the open-source [AutoE2E](https://github.com/autowarefoundation/auto_e2e) end-to-end
driving model on recorded driving data, and visualise what it predicts next to what the
human driver actually did.

AutoE2E takes six surround cameras (plus 3.5 s of camera history), a map raster, the
planned route and the car's recent motion, and outputs the next 6.4 s of driving as
64 steps of acceleration and curvature. This repo loads the released checkpoint, feeds it
scenes from the KITScenes validation set, and draws the result:

<img width="400" height="200" alt="with_route_vs_human_10hz" src="https://github.com/user-attachments/assets/1229947c-8653-45b3-be37-9f4feae02b16" />


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
treat these as observations, not statistics. Every experiment, its method and its full
findings: [`experiments/`](experiments/README.md). Raw numbers and charts: [`results/`](results/).

**1. Accuracy is in line with the authors.** Mean ADE @ 3 s, every moment evaluated:
0.78 m on a 2 s drive (20 moments) and 0.69 m on a 34 s drive (`part-d646a7f28b51e27f`,
340 moments). The authors report 0.744 m on 202 keyframes of the full validation set.
Errors are smallest on straight roads and largest in tight turns (up to 1.85 m).

**2. Route data is missing for 17 of 140 validation drives**, including the 5 longest
(`scripts/tools/scan_routes.py`, [`route_scan.csv`](results/kitscenes_val/route_scan.csv)).
On those drives the model has to guess at junctions; in earlier runs it sometimes took
the wrong branch or roundabout exit.

**3. The route helps a little on average, a lot at turns, and sometimes too much**
(0.69 m with vs 0.76 m without on one drive).
Details: [`experiments/route_on_off`](experiments/route_on_off/README.md).

**4. Speed: ~24 s per moment (~0.04 FPS)** on an RTX 5060 Laptop in FP32, ~2.3 GB GPU
memory. The FPS tables in the AutoE2E repo (19–76 FPS) are for lighter SwinV2/ConvNeXt
configurations without the BEVFormer T8 history, not for the released checkpoint.

## Layout

```
autoe2e_lab/             shared code used by everything
  upstream.py            finds the AutoE2E clone and checks it is at the right commit
  config.py              loads configs/*.yaml (everything dataset-specific)
  data.py                reads scenes ("parts") in the AutoE2E packed (WebDataset) format
  model.py               loads the checkpoint; predict() mirrors the authors' evaluation loop
  render.py              camera projection and drawing
  experiment.py          helpers shared by all experiments (rate, error, summary)
configs/                 one YAML file per dataset
scripts/basics/          learning steps 01-04, run in order
scripts/tools/           helpers: download a scene, scan routes, publish results
experiments/             one folder per question: run.py + README.md (index: experiments/README.md)
outputs/                 everything the scripts write (not committed)
results/                 small tables and charts copied from outputs/ (committed)
```

Three rules keep it tidy:

1. **An experiment compares setups and produces error numbers** → `experiments/<name>/`.
   Everything else → `scripts/`.
2. **One name everywhere:** `experiments/route_on_off/` writes to
   `outputs/<dataset>/route_on_off/` and is published to `results/<dataset>/route_on_off/`.
3. **Paths read dataset → experiment → part:**
   `outputs/kitscenes_val/route_on_off/part-d646a7f28b51e27f/`. Each experiment also writes
   `summary_<rate>.csv` next to its part folders, one row per part plus all parts together.

```
outputs/kitscenes_val/
├── route_scan.csv       list of all drives (from scripts/tools/scan_routes.py)
├── basics/              pictures and videos from scripts/basics/
├── route_on_off/        one experiment: <part>/ folders + summary_<rate>.csv
├── camera_count/
└── sanity_checks/
```

### Where files go: `outputs/` vs `results/`

| | `outputs/` | `results/` |
|---|---|---|
| What | Everything the scripts write: pictures, videos, cached predictions, logs | Small tables (CSV) and charts, no camera images |
| Written by | The scripts, automatically | `scripts/tools/publish_results.py` |
| On GitHub | No (`.gitignore`) | Yes |
| If deleted | Re-run the scripts | Re-run `publish_results.py` |

`outputs/` is too big for GitHub and contains KITScenes camera images, so it stays local.
After an experiment, run `python scripts/tools/publish_results.py` to copy the shareable
files to `results/` with the same folder layout, then commit `results/`. It copies only
files with the standard names `summary_*.csv`, `results_*.csv`, `error_over_time_*.png`
and `average_by_setup_*.png`, so new experiments that use these names are published too.

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

More scenes: `python scripts/tools/download_scene.py --list`, then `python scripts/tools/download_scene.py <part>`.

## Run

```bash
# Basics: learning steps, run in order
python scripts/basics/01_load_checkpoint.py   # weights load with an exact layer match
python scripts/basics/02_inspect_sample.py    # look at the inputs; no model involved
python scripts/basics/03_infer_one_sample.py  # one prediction: error + camera/map picture
python scripts/basics/04_make_video.py --part <part>   # video at 10 Hz (every moment)

# Experiments: one or more parts each
python experiments/route_on_off/run.py --part <part> [<part> ...]   # with vs without route
python experiments/camera_count/run.py --part <part> [<part> ...]   # 6 vs 3 vs 1 vs 0 cameras
python experiments/sanity_checks/run.py --part <part>               # break things on purpose

# Tools
python scripts/tools/download_scene.py <part>   # download one scene
python scripts/tools/scan_routes.py             # which scenes have a route (~2 MB read per scene)
python scripts/tools/publish_results.py         # copy shareable tables/charts to results/
```

`--step N` uses every Nth moment (default 1 = every moment = 10 Hz; `camera_count`
defaults to 10 = 1 Hz because it runs the model 4 times per moment). Output names carry
the rate (`..._10hz.mp4`, `..._1hz.csv`), so runs at different rates never overwrite each
other. Predictions are cached per moment, so a stopped run continues where it left off,
and a 10 Hz run reuses the moments a 1 Hz run already computed.

Long runs (hours) are best started detached from the terminal:

```bash
mkdir -p outputs/logs
nohup python experiments/route_on_off/run.py --part <part> > outputs/logs/<name>.log 2>&1 &
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
