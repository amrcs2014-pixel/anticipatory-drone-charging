# When does anticipation pay? Charging decisions in dynamic drone delivery

Code, instances, schedules and raw results for the manuscript
"When does anticipation pay? Charging decisions in dynamic drone delivery with
fixed and mobile chargers".

Drones recharge at shared fixed stations or have their batteries swapped by a
ground vehicle; the queues they meet are produced by the fleet's own charging
choices. The code plans dynamic delivery days once, then flies the same days
under different charging policies: the regret rule of Fang et al. (2026),
simple rules (nearest, fastest), a rollout ceiling that knows the schedule, and
a learned (LightGBM) estimate of each option's future cost.

## Layout

| Path | Content |
| --- | --- |
| `code/drp/` | Simulator, ALNS routing engine, dynamic days, charging policies, learned policy |
| `code/*.py` | Experiment scripts (see `code/README.md` for every command and result) |
| `code/case/` | Amman case: OpenStreetMap download and instance builder, hourly wind data |
| `code/results/` | Cached schedules, trained models and raw outputs behind every table |
| `code/figures/` | Figures 1 to 5 (PNG and PDF) and the script that draws them |
| `code/tests/` | Regression checks: `python code/tests/test_core.py` |
| `figshare_instances/` | Benchmark instances of Fang et al. (2026), unchanged |

## Requirements

Python 3.11. The simulator and the routing engine use the standard library
only; the learned policy and the figures need

    pip install -r requirements.txt

## Reproducing the paper

Run the scripts from `code/`. The main tables:

    python paper_tables.py

Section 6.6 (Amman case) and Section 6.7 (wind):

    python run_rollout.py AMMAN75_R5 --seeds 330 --ugv 1
    python run_wind.py C75_R5 --first-day 300 --days 100 --model results/models_C75_R5_ugv1_fastest.pkl

Schedules are cached in `code/results/schedules_*.json`, so the tables can be
re-run without re-planning. `code/README.md` lists every experiment, its command
and its output file, including results that were checked and not reported.

## Data and licences

- Benchmark instances: Fang, Huang, Wang, Zhu (2026), Applied Soft Computing,
  https://doi.org/10.6084/m9.figshare.32252883, CC BY 4.0.
- Amman locations: (c) OpenStreetMap contributors, Open Database Licence (ODbL).
  The identifiers of the map objects used are in `code/case/*_sources.json`.
- Wind: ERA5 reanalysis (Hersbach et al., 2020) through the Open-Meteo archive
  API, CC BY 4.0.
- Code: MIT licence (see `LICENSE`).
