# Results

Small result files copied from `outputs/` (which is not committed) by
`scripts/tools/publish_results.py`. Tables and charts only: no camera images, no videos.
The folder layout mirrors `outputs/`: `results/<dataset>/<experiment>/<part>/`.

What each experiment's files mean, and what we found: see its README in
[`experiments/`](../experiments/README.md).

`<rate>` in a file name is how many moments per second of driving were evaluated
(10hz = every moment, 1hz = every 10th). "Error" is ADE @ 3 s: the average distance
between the predicted and the real path over the next 3 s.

| File | What |
|---|---|
| `<dataset>/route_scan.csv` | All drives of the dataset: length, size, route and map available, upcoming manoeuvre, why the route is missing (`scripts/tools/scan_routes.py`) |
| `<dataset>/<experiment>/summary_<rate>.csv` | One row per drive plus all drives together |
| `<dataset>/<experiment>/<part>/results_<rate>.csv` | One row per moment |
| `<dataset>/<experiment>/<part>/*.png` | Charts |
