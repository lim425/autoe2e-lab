# map_on_off: does the model use the map?

## Question

AutoE2E gets a 14-channel semantic map raster (lanes, road edges, crossings, ...). The
model card calls it optional ("it can be left empty"). Does the prediction change without
it, and how much worse does it get?

## Method

For every moment, the model answers twice with the same cameras, route and motion history:

- **with map**: the dataset's map, as normal (green in the video)
- **without map**: the map switched off with the model's own validity gate,
  `map_valid=False`, which multiplies it by 0 (blue)

The route stays on in both. Both answers are compared with the human's real path (purple).
This is the same "off" switch the authors use for their camera-only KITScenes Test track.

The camera top-down picture is built from the cameras only, before the map is mixed in,
so it is computed once per moment and reused for the second answer. Three self-checks run
at the start: reusing the picture changes nothing (a, b), and the gate gives exactly the
same answer as an all-zero map (c).

Limit: the checkpoint was trained with a map (the authors' paper: "trained under a
condition in which the map and the route are provided"). "Without map" is an input it
never practised, so a large error shows it depends on the map, and also that it never
learned to drive without one.

## How to run

```bash
python experiments/map_on_off/run.py --part part-d646a7f28b51e27f
python experiments/map_on_off/run.py --part part-d646a7f28b51e27f part-8028dd4a50d07877
```

Default `--step 1` (every moment, 10 Hz): about 25 s per moment on an RTX 5060 Laptop,
~2.5 h for a 34 s drive. For a long run, start it detached from VS Code:

```bash
nohup python experiments/map_on_off/run.py --part part-d646a7f28b51e27f > outputs/logs/map_on_off_d646_10hz.log 2>&1 &
```

Outputs in `outputs/<dataset>/map_on_off/`:

| File | What |
|---|---|
| `<part>/results_<rate>.csv` | One row per moment: speed, map/route available, manoeuvre, error with and without map, difference (positive = map helped) |
| `<part>/error_over_time_<rate>.png` | Both errors along the drive |
| `<part>/average_by_setup_<rate>.png` | Average error with and without map |
| `<part>/map_on_off_<rate>.mp4` | Camera + map with both paths and the human |
| `<part>/without_map_vs_human_<rate>.mp4` | Only the model without map (blue) vs the human |
| `summary_<rate>.csv` | One row per drive plus all drives together |

## Findings

`part-d646a7f28b51e27f`, every moment (10 Hz, 340 moments), route on in both. Map and
route are available at every moment of this drive.

| | with map | without map |
|---|---|---|
| Mean error, next 3 s | **0.69 m** | 0.83 m (+20 %) |
| Median error | 0.61 m | 0.78 m |
| Moments with error > 1 m | 69 | 102 |

- **The model uses the map, and it helps on most moments.** The map helped (≥ 5 cm
  better) at 69 % of moments, hurt at 13 % and made no difference at 18 %.
- **Its biggest effect is at the start of the right turn** (t = 0.0–0.3 s, 11 km/h,
  roundabout-like junction). With the map the model follows the human's curve (0.25 m).
  Without it, it drives almost straight across the junction (1.40 m). From the cameras
  alone it does not see the road curving away.
- **It matters most at low speed and in turns.** 10–20 km/h: 0.79 vs 1.01 m. 30–40 km/h
  (mostly straight): 0.34 vs 0.44 m.
- **It is not always right.** The worst moment is t ≈ 15 s (left curve along the tram
  tracks, 19 km/h). With the map the model turns too sharply (1.85 m). Without the map
  it does better (1.23 m). The route experiment shows the same moment: with map and route
  1.35 m on average over t = 14.5–15.5 s, without the map 0.98 m, without the route 0.64 m.
  At this spot, map and route together push the model into too tight a turn.
- **The map matters more than the route on this drive.** Removing the map adds 0.14 m on
  average; removing the route adds 0.08 m ([route_on_off](../route_on_off/README.md)).
  The two effects follow each other closely (correlation 0.75 per moment).
- Check: the "with map" numbers are identical to "with route" in route_on_off (same
  setup, max difference 0.0), so runs are reproducible.

Remember the limit above: the model was never trained without a map, so this measures
how much this checkpoint relies on the map. It does not show how well a map-free model
could drive.

Next: repeat on `part-8028dd4a50d07877` (also has map and route) to see if this holds on
a second drive.
