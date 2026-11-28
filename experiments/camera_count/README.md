# camera_count: how much do fewer cameras hurt?

## Question

The released checkpoint always takes 6 cameras. If a vehicle has fewer, how much worse
does the prediction get, and which cameras matter most?

## Method

Missing cameras are replaced by black images: in the current frame, in all 7 past frames,
and for the front camera also in its two extra high-resolution copies. Map, route and
motion history are not changed. A self-check confirms that kept cameras are untouched and
dropped cameras are fully black.

| Setup | Real cameras | Black |
|---|---|---|
| 6 cameras | all | none (the normal model) |
| 3 front | front, front-left, front-right | the 3 rear |
| front only | front | the other 5 |
| no cameras | none | all 6 (what map + route + motion history give alone) |

Limit: the model was never trained with black cameras, and this is open-loop on KIT car
data. It shows how much **this model** leans on each camera group, not how a vehicle with
fewer cameras would really drive.

## How to run

```bash
python experiments/camera_count/run.py --part part-d646a7f28b51e27f
python experiments/camera_count/run.py --part part-d646a7f28b51e27f part-8028dd4a50d07877
```

Default `--step 10` (1 Hz): each setup needs its own full camera pass, so 4 setups take
~1.6 min per moment; a 34 s drive at 1 Hz takes ~55 min. `--step 1` (10 Hz) takes ~9 h.

Outputs in `outputs/<dataset>/camera_count/`:

| File | What |
|---|---|
| `<part>/results_<rate>.csv` | One row per moment: error for every setup |
| `<part>/error_over_time_<rate>.png` | All setups along the drive |
| `<part>/average_by_setup_<rate>.png` | Average error per setup |
| `<part>/biggest_gap_<rate>.png` | Top-down view of the moment where fewer cameras hurt most (local only: shows the map) |
| `summary_<rate>.csv` | One row per drive plus all drives together |

## Findings

Not run in the current layout yet.

To re-check: an earlier exploratory run (1 drive, 34 moments) gave 0.68 m with 6 cameras
and 0.76 m for the other three setups, with "front only" equal to "no cameras". Looking
inside, the planner read its 16 sample points per step only from the rear-right corner of
the top-down picture (25–60 m behind the car), where the front cameras have no effect.
