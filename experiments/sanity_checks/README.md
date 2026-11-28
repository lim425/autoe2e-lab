# sanity_checks: does the setup react as it should when broken on purpose?

## Question

Can the results be trusted? If the setup were wrong (or the green line faked), breaking
the inputs on purpose would not change the results the way it should.

## Method

At a few moments of one drive, run these tests:

| Test | What changes | Expected |
|---|---|---|
| normal | nothing: trained model, real inputs | small error |
| run again | identical second run | exactly the same answer |
| empty brain | same model structure, random weights | nonsense |
| black cameras | every camera image black | worse |
| cameras shuffled | the 6 camera images in a different order | worse |
| first step | first predicted point vs speed × 0.1 s | about equal |

## How to run

```bash
python experiments/sanity_checks/run.py                                   # 3 moments, part-d646
python experiments/sanity_checks/run.py --part part-xxxx --moments 0 50 100
```

About 2 min per moment. Outputs in `outputs/<dataset>/sanity_checks/<part>/`:

| File | What |
|---|---|
| `results_checks.csv` | Error of every test at every moment |
| `<moment>.png` | Top-down view with every test's prediction (local only: shows the map) |

## Findings

`part-d646a7f28b51e27f`, moments 0, 100, 200 (average error over the next 3 s):

| Test | Error |
|---|---|
| normal | 0.56 m |
| run again | identical (bit for bit) |
| empty brain | 4.40 m (drives in circles, off the road) |
| black cameras | 0.72 m |
| cameras shuffled | 0.66 m |
| first step | equals speed × 0.1 s exactly |

- **The setup is sound:** repeat runs are identical, random weights give nonsense, and the
  first step matches the current speed.
- **Cameras matter less than expected:** with all cameras black the error only rises from
  0.56 m to 0.72 m, so the planner leans on the map, the route and the motion history.
  A known limit of open-loop evaluation: low error does not prove the model reacts to
  what only cameras can see.
