# Experiments

One folder per question. Each folder has `run.py` (the experiment) and `README.md`
(question, method, how to run, findings). All errors are ADE @ 3 s, open-loop: the average
distance between the predicted and the human's real path over the next 3 s.

| Experiment | Question | Status | Main result |
|---|---|---|---|
| [`route_on_off`](route_on_off/README.md) | Does the model use the route? | ✅ done: 1 drive, 10 Hz | 0.69 m with route vs 0.76 m without |
| [`sanity_checks`](sanity_checks/README.md) | Does the setup react as it should when broken on purpose? | ✅ done: 1 drive, 3 moments | all cameras black: 0.56 → 0.72 m |
| [`map_on_off`](map_on_off/README.md) | Does the model use the map? | ✅ done: 1 drive, 10 Hz | 0.69 m with map vs 0.83 m without |

## Where everything is

For an experiment called `<name>`:

| What | Where |
|---|---|
| Code and explanation | `experiments/<name>/run.py`, `experiments/<name>/README.md` |
| Everything it produced (laptop only) | `outputs/<dataset>/<name>/<part>/` |
| All parts together | `outputs/<dataset>/<name>/summary_<rate>.csv` |
| Shareable tables and charts (GitHub) | `results/<dataset>/<name>/` |

## Adding a new experiment

1. Create `experiments/<name>/run.py`. Use the helpers in `autoe2e_lab/experiment.py`:
   `part_args` (`--part` one or more, `--step`), `rate_tag`, `ade3`, `experiment_dir`
   and `write_summary`. `route_on_off/run.py` is a complete example.
2. Write into `outputs/<dataset>/<name>/<part>/` and use the standard file names, so
   `scripts/tools/publish_results.py` picks them up:
   `results_<rate>.csv` (one row per moment), `error_over_time_<rate>.png`,
   `average_by_setup_<rate>.png` and, from `write_summary`, `summary_<rate>.csv`.
3. Copy `README.md` from another experiment and fill in the four sections.
4. Add a row to the table above.
5. Run it, then `python scripts/tools/publish_results.py` and commit.
