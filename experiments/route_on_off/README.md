# route_on_off: does the model use the route?

## Question

AutoE2E gets a route raster (the corridor it should follow, plus the destination). Does
its prediction actually change when the route is there, and does it get better?

## Method

For every moment, the model answers twice with the same cameras, map and motion history:

- **with route**: the dataset's route, as normal (green in the videos)
- **without route**: the route switched off with the model's own validity gate,
  `route_valid=False`, the authors' "route-zero" mode (orange)

Both answers are compared with the human's real path (purple). The camera top-down picture
is computed once and reused for the second answer, so the experiment costs about the same
as a normal run. A self-check at the start confirms that the reuse changes nothing.

Only moments where the dataset has a valid route can show a route effect. Use
`scripts/tools/scan_routes.py` to pick drives that have one. The route in KITScenes is an
oracle reconstructed from the whole recorded drive, as the authors note in their paper.

## How to run

```bash
python experiments/route_on_off/run.py --part part-d646a7f28b51e27f                  # one drive
python experiments/route_on_off/run.py --part part-d646a7f28b51e27f part-8028dd4a50d07877  # several
```

Default `--step 1` (every moment, 10 Hz): about 24 s per moment on an RTX 5060 Laptop.

Outputs in `outputs/<dataset>/route_on_off/`:

| File | What |
|---|---|
| `<part>/results_<rate>.csv` | One row per moment: speed, route available, manoeuvre, error with and without route, difference (positive = route helped) |
| `<part>/error_over_time_<rate>.png` | Both errors along the drive |
| `<part>/route_on_off_<rate>.mp4` | Camera + map with all three paths |
| `<part>/with_route_vs_human_<rate>.mp4` | Only the model with route vs the human |
| `summary_<rate>.csv` | One row per drive plus all drives together |

## Findings

| Drive | Moments | With route | Without route | Route helped / hurt / same |
|---|---|---|---|---|
| `part-d646a7f28b51e27f` (34 s, right turn, 10 Hz) | 340 | **0.69 m** | 0.76 m | 46% / 31% / 23% |

- **The route clearly changes the output** and helps most at turns where there is a
  choice: up to 1.16 m better in the right turn at the start of the drive.
- **It can also hurt:** in a tight left turn (t ≈ 15 s) the route made the model turn too
  sharply (1.85 m with route vs 0.81 m without).
- **One drive is not enough** to say how big the average gain really is. The authors'
  paper notes that this paired with/without comparison is not yet published.
