# Results

Small result files copied from `outputs/` (which is not committed). No camera images.

- `route_experiment/<part>/results_<rate>.csv`: one row per moment. Columns: time in the
  drive, speed, whether the dataset has a route, upcoming manoeuvre, intersection flag,
  ADE @ 3 s with route, ADE @ 3 s without route, and the difference (positive = route helped).
- `route_experiment/<part>/error_over_time_<rate>.png`: both errors along the drive.

`<rate>` is how many moments per second of driving were evaluated (10hz = every moment,
2hz = every 5th). Regenerate with `scripts/05_route_experiment.py --part <part> --step 1|5`.
