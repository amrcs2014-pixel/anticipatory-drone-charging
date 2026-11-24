# Anticipatory charging decisions in dynamic drone delivery: code

Phase 1 of the work plan: a simulator and routing engine for the base model of
Fang et al. (2026), with a hook for the learned charging policy.

## Layout

| File | Purpose |
| --- | --- |
| `drp/params.py` | All model parameters; each is tagged `[paper]`, `[assumed]` or `[calibrated]` |
| `drp/instance.py` | Loader for the Figshare instance files |
| `drp/simulate.py` | Event-driven simulation of one day: energy, fuzzy satisfaction, station queues, UGV swaps |
| `drp/policies.py` | Charging rules: regret (paper Eq. 1), cost-based (Eq. 11), nearest, station-only wrapper, and the anticipatory hook |
| `drp/alns.py` | ALNS routing engine: 6 destroy and 3 repair operators, score-based adaptive weights |
| `run_static.py` | Solve one instance from the command line |
| `calibrate_units.py` | Tests distance scales against the paper's Table 6 |
| `drp/dynamic.py` | Rolling-horizon day: order release, re-planning, dispatch timing |
| `run_dynamic.py` | Dynamic days on an energy-binding version of an instance |
| `tests/test_core.py` | 11 checks; run with `python tests/test_core.py` |

Python 3.11, standard library only.

## What has been verified

- The regret and weighted-sum scores reproduce the paper's worked example in
  Section 3.2 (0.80 / 0.51 / 0.69 and 0.60 / 0.51 / 0.40).
- With the future-cost term set to zero, the anticipatory policy makes the same
  choice as the regret rule and as the cost rule (tested).
- A hand-computed two-customer route gives the expected distance, cost and
  arrival energy.
- Shared pads produce waiting when two drones reach a one-pad station together.

## Unit calibration

The files do not state units. Taking each small instance of the paper's
Table 6 as the first n customers of the matching file, a scale of
**0.1 km per coordinate unit** gives costs within about 1% of the published
values, and two exact matches:

| Instance | This code (scale 0.1) | Paper, Table 6 |
| --- | --- | --- |
| R12_R4 (from R75_R4) | 335.05 | 335.05 |
| C15_R4 (from C75_R4) | 428.92 | 428.78 |
| C7_R5 (from C100_R5) | 212.24 | 212.67 |
| RC10_R6 (from RC100_R6) | 332.49 | 333.01 |
| RC12_R4 (from RC75_R4) | 438.19 | 439.82 |
| R11_R3 (from R75_R3) | 329.76 | 329.76 |
| RC6_R3 (from RC100_R3) | 218.67 | 220.33 |
| C11_R3 (from C100_R3) | 322.97 | 324.64 |

How the small instances were built is itself an assumption, so this is strong
evidence for the scale, not proof.

## Full instances: why the costs are below the published ones

| Instance | This code, 3 runs | Same model, paper-like search | Paper EALNS, mean / best |
| --- | --- | --- | --- |
| C75_R5 | mean 2469.56, best 2409.68 | mean 2563.44 | 2703.99 / 2656.25 |
| R75_R3 | mean 2175.39, best 2174.46 | mean 2260.06 | 2427.41 / 2392.71 |
| RC75_R4 | not run | mean 2371.77 | 2496.97 / 2418.43 |

"Paper-like search" = start temperature 3000, cooling 0.99, and only the paper's
three removal and two insertion operators.

Best-supported explanation: the model is equivalent where it can be checked,
and the published search is far from optimal at this size. Evidence:

1. Two small instances with Gurobi values are reproduced to the cent,
   satisfaction penalties included: R11_R3 = 329.76 and R12_R4 = 335.05
   (now a regression test).
2. Those optimal routes use 55 to 68 Wh of the 100 Wh battery, so the paper's
   model also accepts such routes without charging.
3. The paper's Figure 4 shows its search on C75_R5 starting near 2950 and
   improving only about five times in 400 iterations.
4. With a start temperature of 3000 the search accepts a solution 100 worse
   with probability above 95% for most of the run; switching this code to
   those settings raises its costs by about 4%, part of the way to the
   published values.

What remains unexplained: about 5 to 7% between the paper-like run and the
published means. It cannot be settled without the authors' code, because no
exact solver handles 75 customers. Do not present the lower costs as a
contribution until the authors confirm the model reading.

## Consequence for the research design

At the calibrated scale the map is 10 km wide and a drone's range is 14 to
24 km, so **good routes never need to charge**: 0 charging events in every
near-optimal solution on the base instances. A study of charging decisions
needs instances where energy binds. Sweep on C75_R5 (150 iterations, one seed):

| Setting | Cost | Routes | Station visits | Swaps | Feasible |
| --- | --- | --- | --- | --- | --- |
| scale 0.10 (base) | 2409.68 | 22 | 0 | 0 | yes |
| scale 0.15 | 2597.11 | 23 | 1 | 0 | yes |
| scale 0.20 | 2694.13 | 23 | 2 | 4 | yes |
| scale 0.25 | 2875.30 | 23 | 10 | 10 | yes |
| scale 0.30 | 4723.58 | 29 | 13 | 15 | no (1 late) |
| battery 60 Wh | 2497.39 | 23 | 0 | 1 | yes |
| battery 45 Wh | 2543.19 | 23 | 7 | 4 | yes |

A map about 2.5 times wider (scale 0.25), or a 45 Wh battery, gives roughly
10 to 20 charging decisions per day while staying feasible. The dynamic
experiments should be built in that regime.

## Open assumptions

| Item | Value used | Status |
| --- | --- | --- |
| Distance scale | 0.1 km per unit | calibrated, see above |
| Time unit | minutes | assumed |
| Charging rate | 80 Wh per minute | assumed reading of "80w per minute" |
| External queue (M/G/1) parameters | lambda 0.05, mean 0.6, variance 0.1 | assumed; not in the paper |
| UGV speed | 15 m/s | midpoint of the paper's 10 to 20 m/s |
| Early arrival | service starts at the tolerable earliest time | reading of Eq. 29 |

## Simplifications against the base paper

- Charging is reactive: decided only when the next leg plus a reserve to the
  nearest refuge is not covered.
- UGVs are dispatched on request, not on a pre-planned tour.
- The initial solution is a greedy insertion, not the paper's ant colony step.
- Simulated-annealing start temperature is set from the initial cost, not 3000.

## Dynamic operation (`drp/dynamic.py`, `run_dynamic.py`)

Parcels are loaded at the depot, so a drone in the air cannot take new orders.
The dynamic decisions are: which known orders share a route (ALNS, re-run when
orders arrive, warm-started from the previous plan), when each route leaves
(held while leaving later costs nothing), and where each drone recharges
(the charging rule, inside one shared day simulation).

Two additions made while building it:

- `filter_late` (default on): no rule may pick an option that makes the next
  customer late when another option avoids it. Applied to every rule alike.
- Late service costs 200 per order in dynamic runs (`--late-penalty`), instead
  of the 1000 used as a hard-constraint penalty in static runs.

Rules are compared on identical days: routes and departure times are fixed
first with a planning rule (`plan_day`), then every charging rule flies that
same schedule (`execute`).  The perfect-information benchmark uses the same
procedure with all orders known at time 0 and a 200-iteration first plan; it
takes the better of its own schedule and the dynamic one, so it can never be
worse than the dynamic run.

C75_R5 at scale 0.25, one pad per station, 50% of orders revealed during the
day, 10 seeds, 20 ALNS iterations per re-plan (raw output in `results/`).
Mean cost with 95% confidence half-width:

| UGVs | Rule | Dynamic cost | Late orders | Swap wait, min | Perfect-information |
| --- | --- | --- | --- | --- | --- |
| 1 | regret | 3672.96 (495.06) | 3.6 | 221.7 | 3234.35 |
| 1 | cost | 3672.96 (495.06) | 3.6 | 221.7 | 3234.35 |
| 1 | nearest | 3569.32 (436.83) | 3.1 | 207.1 | 3225.88 |
| 2 | regret | 2918.92 (169.55) | 1.2 | 60.4 | 2642.47 |
| 2 | cost | 2919.12 (169.51) | 1.2 | 60.2 | 2642.38 |
| 2 | nearest | 2886.51 (177.28) | 0.8 | 36.9 | 2514.01 |

Paired differences (same schedule, same seed):

| UGVs | Comparison | Mean | 95% CI | Verdict |
| --- | --- | --- | --- | --- |
| 1 | cost minus regret | 0.00 | [0.00, 0.00] | identical on all 10 days |
| 1 | nearest minus regret | -103.64 | [-422.90, +215.62] | not significant |
| 2 | cost minus regret | +0.20 | [-0.25, +0.65] | not significant, tied on 9 days |
| 2 | nearest minus regret | -32.40 | [-254.45, +189.64] | not significant |

Value of information (dynamic minus perfect-information):

| UGVs | Rule | Mean | 95% CI | Share of benchmark |
| --- | --- | --- | --- | --- |
| 1 | regret | +438.61 | [+70.25, +806.98] | 13.6% |
| 1 | nearest | +343.43 | [+120.01, +566.86] | 10.6% |
| 2 | regret | +276.44 | [+96.48, +456.41] | 10.5% |
| 2 | nearest | +372.51 | [+129.16, +615.86] | 14.8% |

What this shows:

- The three myopic rules cannot be told apart once they fly the same schedule.
  The large gaps seen in earlier runs were mostly planning noise.
- The regret rule and the cost rule make almost the same choices here.
- Not knowing the orders in advance costs 10 to 15%, significantly above zero.
- A second UGV lowers cost by about 20% and cuts swap waiting by more than
  two thirds.

What it does not show:

- The benchmark is valid by construction but weak: the perfect-information
  plan lost to the dynamic schedule on 9 of 30 (one UGV) and 8 of 30 (two
  UGVs) rule-days, so the true value of information is probably larger.
- The 10 to 15% gap includes batching and departure timing, not only
  charging.  How much of it a better charging decision can recover is unknown.
- One instance, one scale, one dynamism level.

## Go / no-go test: rollout (`run_rollout.py`)

Question: does the charging decision have enough leverage to justify a
learned policy?  At every charging decision with more than one option, rollout
tries each option, plays out the rest of the day with the regret rule, and
keeps the cheapest.  `rollout-air` simulates only drones already in the air;
`rollout-all` also simulates later scheduled departures, which uses knowledge
of the future and is therefore an optimistic ceiling.

C75_R5, scale 0.25, one pad per station, 50% dynamic orders, 30 days, all
rules on the same schedules (raw output and cached schedules in `results/`).

| UGVs | Rule | Mean cost | Late orders | Difference against regret | 95% CI | Days better / worse |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | regret | 3797.00 | 4.33 | | | |
| 1 | nearest | 3552.53 | 3.10 | -244.47 (-6.4%) | [-415.62, -73.32] | 16 / 11 |
| 1 | rollout-air | 3827.62 | 4.47 | +30.61 (+0.8%) | [-221.33, +282.56] | 16 / 10 |
| 1 | rollout-all | 3190.68 | 1.30 | -606.32 (-16.0%) | [-804.70, -407.94] | 28 / 0 |
| 2 | regret | 2963.28 | 1.23 | | | |
| 2 | nearest | 2883.61 | 0.73 | -79.67 (-2.7%) | [-184.39, +25.04] | 9 / 20 |
| 2 | rollout-air | 2882.56 | 0.83 | -80.72 (-2.7%) | [-162.37, +0.93] | 25 / 5 |
| 2 | rollout-all | 2789.83 | 0.37 | -173.46 (-5.9%) | [-265.09, -81.82] | 29 / 0 |

Per day there are about 22 to 23 charging decisions; 12 (one UGV) or 20 (two
UGVs) of them have more than one option; rollout changes only 3 to 4 of them.

Reading:

- **Go.**  With knowledge of later departures, changing about 3 charging
  decisions a day lowers cost by 16% (one UGV) and 6% (two UGVs), both
  significant, and it never does worse than the regret rule.  The charging
  decision has leverage.
- **The leverage comes from anticipation.**  Looking only at drones already in
  the air gains nothing with one UGV and an insignificant 2.7% with two.  The
  gain appears only when future departures are taken into account, which is
  what the learned estimate is meant to predict.
- **This is a ceiling, not a forecast.**  `rollout-all` knows the future
  schedule exactly, including orders not yet released.  A learned policy can
  only know it in expectation, so it will recover part of the gain; how much
  is not known.
- Leverage shrinks as capacity grows (16% to 6% with a second UGV).
- With 30 days the nearest rule is significantly better than the regret rule
  with one UGV (-6.4%), which 10 days could not show.
- One instance, one scale, one dynamism level.

### Across twelve settings (`results/rollout_summary.txt`, `summarize_rollout.py`)

30 days each, same procedure.  Percentages are relative to the regret rule;
`*` = significant at 5%.

| Instance | Scale | Battery | UGVs | Dynamic share | Regret cost | Late | nearest | rollout-air | rollout-all | 95% CI, rollout-all |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C75_R5 | 0.25 | 100 | 1 | 25% | 3771.0 | 4.13 | -8.0%* | -5.0% | -16.6%* | [-888, -361] |
| C75_R5 | 0.25 | 100 | 1 | 50% | 3797.0 | 4.33 | -6.4%* | +0.8% | -16.0%* | [-805, -408] |
| C75_R5 | 0.25 | 100 | 1 | 75% | 3781.8 | 4.47 | -7.1%* | -4.2% | -16.8%* | [-886, -381] |
| C75_R5 | 0.25 | 100 | 2 | 50% | 2963.3 | 1.23 | -2.7% | -2.7% | -5.9%* | [-265, -82] |
| C75_R3 | 0.25 | 100 | 1 | 50% | 3248.7 | 2.13 | -3.8% | -3.4%* | -8.0%* | [-397, -124] |
| RC75_R4 | 0.25 | 100 | 1 | 50% | 4071.1 | 5.57 | -4.7%* | -7.9%* | -13.5%* | [-739, -362] |
| RC75_R4 | 0.25 | 100 | 2 | 50% | 3239.4 | 2.23 | -2.7% | +0.7% | -7.7%* | [-347, -149] |
| RC100_R5 | 0.25 | 100 | 1 | 50% | 6480.4 | 10.90 | -2.0% | -1.8% | -10.6%* | [-933, -441] |
| R75_R3 | 0.25 | 100 | 1 | 50% | 2619.2 | 0.50 | -1.0% | -0.1% | -2.7%* | [-111, -32] |
| R75_R3 | 0.25 | 100 | 2 | 50% | 2529.6 | 0.50 | +0.1% | -2.2% | -3.2%* | [-146, -19] |
| R100_R4 | 0.25 | 100 | 1 | 50% | 3861.8 | 0.83 | -0.8% | -0.2% | -3.3%* | [-188, -68] |
| C75_R5 | 0.10 | 45 | 1 | 50% | 2597.3 | 0.43 | -0.3% | -3.2% | -3.2% | [-210, +45] |

- `rollout-all` is significantly better than the regret rule in 11 of 12
  settings and was never worse on any single day.  The gain is 5% or more in 8.
- The gain follows congestion: 8 to 17% where the regret rule leaves 2 to 11
  orders late; about 3% where lateness is rare (R instances, small battery).
- The share of dynamic orders (25%, 50%, 75%) makes no difference.
- `rollout-air`, which uses only information a real controller has, is
  significant in 2 of 12 settings.
- The nearest rule beats the regret rule significantly in 4 of 12 and is never
  significantly worse.
- Shrinking the battery on the original map gives no significant gain;
  widening the map does.  The idea applies where detours are long and
  charging capacity is scarce, and the paper must state that scope.
- RC100_R5 leaves about 11 of 100 orders late, so one UGV at this scale is an
  under-capacity system for 100 customers.

## Learned future-cost estimate (`drp/learn.py`, `train_policy.py`, `eval_policy.py`)

- **Label.** On training days, rollout with future knowledge gives the final
  cost of every charging option; the label of an alternative is that cost
  minus the cost of the regret rule's choice.
- **Inputs.** 27 features available to a real controller at decision time
  (the option, the drone, the fleet in the air, charger state, time of day).
  Nothing about later departures.
- **Model.** Gradient-boosted regression (LightGBM, Huber loss).
- **Decision.** Switch from the regret choice to the best alternative only if
  the predicted saving exceeds a margin.  A zero estimate reproduces the
  regret rule exactly (tested).
- **Data.** Labels along the regret rule's days, the rollout's days, and one
  round along the learned policy's own days.
- **Protocol.** Trained on C75_R5 with one UGV: 120 days for training, 30 to
  choose the margin, 30 held out.  The held-out days and every other setting
  below were never used for training or tuning.

Held-out days of the training setting (30 days):

| Rule | Mean cost | Late orders | Difference against regret | 95% CI |
| --- | --- | --- | --- | --- |
| regret | 3797.00 | 4.33 | | |
| nearest | 3552.53 | 3.10 | -244.47 (-6.4%) | [-415.62, -73.32] |
| rollout-air | 3827.62 | 4.47 | +30.61 (+0.8%) | [-221.33, +282.56] |
| **learned** | **3388.45** | **2.20** | **-408.56 (-10.8%)** | **[-603.41, -213.70]** |
| rollout-all (ceiling) | 3190.68 | 1.30 | -606.32 (-16.0%) | [-804.70, -407.94] |

The learned policy recovers 67% of the ceiling's gain and is also
significantly better than the nearest rule (-164.1, CI [-307.6, -20.5]).

The same model applied unchanged to other settings (30 days each;
`results/transfer_model_C75_R5_ugv1.txt`; `*` = significant at 5%):

| Setting | Learned against regret | Share of ceiling recovered | Learned against nearest |
| --- | --- | --- | --- |
| C75_R5, 1 UGV, 25% dynamic | -11.5%* | 69% | -3.5%* |
| C75_R5, 1 UGV, 75% dynamic | -11.8%* | 70% | -4.7% |
| C75_R5, 2 UGVs | -3.2%* | 54% | -0.5% |
| C75_R3, 1 UGV | -4.5%* | 56% | -0.7% |
| RC75_R4, 1 UGV | -8.3%* | 62% | -3.6% |
| RC75_R4, 2 UGVs | -4.3%* | 56% | -1.6% |
| RC100_R5, 1 UGV | -7.4%* | 70% | -5.4% |
| R75_R3, 1 UGV | -1.7%* | 60% | -0.7% |
| R75_R3, 2 UGVs | -2.2% | | |
| R100_R4, 1 UGV | -1.7%* | 52% | -0.9% |
| C75_R5, original map, 45 Wh | -2.5% | 78% | -2.2% |

Reading:

- Against the regret rule the learned policy is significantly better in the
  training setting and in 9 of the 11 transfer settings, recovering 52 to 78%
  of the ceiling.  It transfers to other customer patterns, 100 customers,
  a second UGV and other dynamism levels without retraining.
- Against the nearest rule its advantage is smaller (0.5 to 5.4%) and
  significant only in the training setting and one transfer setting.  A large
  part of the gain over the regret rule is also available to a trivial rule.
- The learned policy is not uniformly better: it is worse than the regret
  rule on 8 to 24 of 30 days depending on the setting.  The average gain comes
  from days where it avoids late orders.
- All instances derive from the same Figshare family with a widened map; the
  schedules are planned with the regret rule; one training run, no repeats
  over training seeds.

### Against stronger simple rules (supersedes the table above for claims)

The first model was trained to deviate from the regret rule.  Two checks
followed: train it to deviate from a better simple rule, and compare with
the simplest rule that uses the same information, **fastest** (least time
cost: detour or UGV travel, plus waiting and charging).

Training setting, C75_R5 with one UGV: 210 training days, 30 to choose the
margin, **60 held-out days**, 5 independent model fits
(`results/train_fastest_C75_R5_ugv1.txt`):

| Rule | Mean cost | Late orders |
| --- | --- | --- |
| regret (base paper) | 3795.88 | 4.42 |
| nearest | 3570.45 | 3.27 |
| fastest | 3434.53 | 2.45 |
| learned, deviating from fastest | 3291.95 | 1.78 |
| rollout with future knowledge (ceiling) | 3085.50 | 0.83 |

| Paired difference | Mean | 95% CI |
| --- | --- | --- |
| learned minus fastest | -142.58 (-4.2%) | [-243.60, -41.56], significant |
| learned minus nearest | -278.50 (-7.8%) | [-391.66, -165.34], significant |
| learned minus regret | -503.93 (-13.3%) | [-672.50, -335.35], significant |
| ceiling minus fastest | -349.03 (-10.2%) | [-497.33, -200.72], significant |

All five fits are individually significant against the fastest rule
(-132 to -172); the policy changes about 1.5 decisions a day and recovers 41%
of the ceiling's gain over the fastest rule.

The same five models applied unchanged to 11 other settings
(`results/transfer_models_fastest.txt`): better than the fastest rule on
average in 8, significantly in 1 (RC100_R5, -7.1%), never significantly
worse.  Against the nearest rule: significantly better in 6 of 11.

Where the gain over the base paper's rule comes from, in the training setting:

| Step | Cost | Gain |
| --- | --- | --- |
| regret to fastest (a myopic rule that counts waiting) | 3795.88 to 3434.53 | 361.35 (9.5%) |
| fastest to learned (anticipation) | 3434.53 to 3291.95 | 142.58 (4.2%) |
| learned to ceiling (not reached) | 3291.95 to 3085.50 | 206.45 |

### Distilled tree (`distill_policy.py`)

- From the model that deviates from the nearest rule, a depth-2 tree keeps 72%
  of the gain (-224.11 against nearest, CI [-348.71, -99.51]).  Its rule is:
  when the nearest option is the UGV and another option is within about 2
  minutes of the fastest, take that one.  This is close to the fastest rule.
- From the model that deviates from the fastest rule, no tree of depth 2 to 4
  beats the fastest rule (differences +8.73 to +88.28, none significant).
  The extra 4.2% is not captured by a small tree.
- Tree depths were compared on the test days, so the depth-3 figure in
  `results/distill_C75_R5_ugv1.txt` is not a clean estimate.

### What this means

1. In congested dynamic settings the base paper's regret rule is dominated
   by a one-line rule (fastest option).  Most of the gain is there.
2. A learned anticipatory estimate adds a further 4.2% in the setting it is
   trained on, significant and stable over model fits.
3. That extra gain does not transfer reliably without retraining.
4. Knowledge of the future would add about another 6%, so the learned
   estimate captures under half of what anticipation could give.

### Training in a second setting (`results/train_fastest_RC75_R4_ugv1.txt`)

RC75_R4 with one UGV, 120 training days, 30 to choose the margin, 30 held out,
five fits: learned 3472.22 against fastest 3564.23, a difference of -92.01
(-2.6%), 95% CI [-207.36, +23.35], **not significant**; 37% of the ceiling
recovered.  The model trained on C75_R5 gave -109.3 [-244.4, +25.7] on the
same 30 days.  So training in the setting does not clearly help, and 30 test
days cannot detect an effect of this size (the interval is about 115 wide on
each side).  The anticipatory gain is established in one setting only.

### Confirmation with 120 held-out days (`results/train_fastest_RC75_R4_ugv1_400days.txt`)

RC75_R4 with one UGV, 250 training days, 30 for the margin, **120 held-out
days**, five fits: learned 3498.14 against fastest 3608.31, a difference of
-110.17 (-3.1%), 95% CI [-169.93, -50.41], significant in each fit
(-101.25 to -116.03); 54% of the ceiling recovered.  The C75_R5 model applied
unchanged to the same 120 days: -74.9 (-2.1%), CI [-131.5, -18.3], also
significant.  The anticipatory gain is therefore confirmed in a second
setting, and the earlier non-significant result there was a matter of too few
test days.

### Third setting, 100 held-out days (`results/train_fastest_RC100_R5_ugv1_250days.txt`)

RC100_R5 with one UGV, 120 training days, 30 for the margin, 100 held-out
days, five fits: learned 5365.56 against fastest 5675.81, a difference of
-310.25 (-5.5%), 95% CI [-458.82, -161.68], significant in each fit
(-302.25 to -323.81); 90% of the ceiling recovered, with fewer than one
decision changed per day.  The C75_R5 model applied unchanged to the same 100
days: -279.1 (-4.9%), CI [-424.2, -134.0], significant.

Summary of the learned policy trained in the setting, against the fastest rule:

| Setting | Test days | Difference | 95% CI | Ceiling recovered |
| --- | --- | --- | --- | --- |
| C75_R5, 1 UGV | 60 | -4.2% | [-243.60, -41.56] | 41% |
| RC75_R4, 1 UGV | 120 | -3.1% | [-169.93, -50.41] | 54% |
| RC100_R5, 1 UGV | 100 | -5.5% | [-458.82, -161.68] | 90% |

### Clean re-test of C75_R5 on fresh days (`results/fresh_days_C75_R5_ugv1.txt`)

Several model variants had been compared on the 60 held-out days of C75_R5,
so the 4.2% measured there was open to selection.  120 further days (seeds
300 to 419) were planned after the design was fixed and the saved five-fit
model was run on them once:

| Rule | Mean cost | Late orders |
| --- | --- | --- |
| regret | 3903.83 | 4.97 |
| nearest | 3706.10 | 3.97 |
| fastest | 3454.73 | 2.52 |
| learned | 3367.43 | 2.15 |
| ceiling | 3135.06 | 1.09 |

Learned minus fastest: -87.30 (-2.5%), 95% CI [-169.1, -5.5], significant;
four of five fits significant on their own, the fifth narrowly not
(-85.10, CI [-171.40, +1.19]); 27% of the ceiling recovered.  **2.5% is the
figure to report for C75_R5; the 4.2% was inflated by selection.**  The
summary table above therefore reads 2.5% / 27% for C75_R5 on 120 days, and
the claim "all fifteen fits significant" becomes 14 of 15.

### Figures and reference check

- `make_figures.py` writes three figures (PNG at 300 dpi and PDF) to
  `figures/`: the service area, gain against lateness, paired differences.
- Four references written from memory were checked against Crossref and
  OpenAlex on 18 November 2026: Bertsekas et al. (1997) and Powell (2011)
  confirmed; Wang et al. is 2025, volume 9(1), pages 131-144; Reijnen et al.
  was published at ICAPS 2024.  Ke et al. (2017) and Ross et al. (2011) could
  not be confirmed online and remain to be checked.

### Forecast features (`--forecast`, negative result)

Adding features on orders already known but not yet dispatched (count, count
with windows opening within 30 and 60 minutes, count near the option) gave
-113.02 (-3.3%) on the C75_R5 test days instead of -142.58 (-4.2%), with
larger differences between fits.  Not used in the reported model.

### Manuscript tables (`paper_tables.py`, `results/paper_tables.md`)

All rules on the same days in all twelve settings, with the C75_R5 model
applied unchanged outside its training setting.

## Not implemented yet

- A real-world case; all instances derive from one benchmark family.
- Features that describe expected future departures explicitly.
- A stronger perfect-information planner.
- Partial and nonlinear charging; UGV costs in the objective are present but zero by default.

## Usage

    python tests/test_core.py
    python run_static.py C75_R5 --verbose
    python run_static.py R100_R4 --policy cost --runs 10
    python run_static.py C75_R5 --pads 1 --ugv 2
    python calibrate_units.py
    python run_dynamic.py C75_R5 --dod 0 0.5 --seeds 5 --ugv 2

## Reproduction check and submission files (18 November 2026)

- `python tests/test_core.py`: 13/13 pass. `paper_tables.py` re-run gives a file identical to
  `results/paper_tables.md` (`results/paper_tables_rerun.md`). The three held-out evaluations
  (`results/rerun_eval.txt`) reproduce the manuscript's numbers. This re-executes the cached
  schedules and saved models; the schedules were not re-planned and the models were not retrained.
- Figure 4 (`figures/fig4_one_day`): `python make_figures.py --only4`.
- Related work was checked against the Scopus abstracts of the cited papers (not the full texts).
- `../submission/`: highlights, cover letter draft, draft email to the base-paper authors (not sent).
- `../Manuscript_When_does_anticipation_pay.docx`: Word export of the manuscript.

## Case on real locations: Amman (18 November 2026)

- `case/fetch_osm.py` downloads pharmacies, hospitals and fuel stations from OpenStreetMap
  (`case/osm_amman.json`); `case/build_case.py [N]` builds `case/AMMAN75_R5.txt` and
  `case/AMMAN150_R5.txt` (+ `_sources.json` with the OSM ids). Real: locations. Synthetic: time
  windows and demands (from C75_R5, and R75_R3 for the larger case), straight-line travel.
  Coordinates are in units of 0.25 km, so every script runs with its defaults.
- `run_rollout.load_instance` looks in `case/` first.
- Runs: `run_rollout.py AMMAN75_R5 --seeds 330`, `train_policy.py AMMAN75_R5 --days 330 --test 100
  --val 30`, `eval_policy.py results/models_C75_R5_ugv1_fastest.pkl AMMAN75_R5 --days 100`,
  `run_rollout.py AMMAN150_R5 --seeds 60`, and `paper_tables.py <model> "AMMAN75_R5,1,0.5,0.25,100,100;AMMAN150_R5,1,0.5,0.25,100,60"`
  (outputs in `results/case_*`).
- Result: the case is uncongested (under 0.2 late orders a day). Ceiling vs fastest -0.9% (75) and
  -0.7% (150), both significant; simple rules do not differ; the learned policy trained in the case
  gives +0.1% [-25.3, +32.3], not significant. The case confirms the boundary of the main finding,
  not the gain.
- Figure 5: `python make_figures.py --case`.

## Robustness to wind (`run_wind.py`, Section 6.7)

Wind: `case/wind_amman_2025.json`, hourly wind at 100 m for Amman, 2025 (ERA5 through the
Open-Meteo archive API, CC BY 4.0).  The drone holds an airspeed of 22 m/s; a leg's time and
energy are multiplied by airspeed / ground speed.  Draws with more than 12 m/s in the five
flying hours are redrawn.  An in-flight battery depletion is penalised like a late order.

    python run_wind.py AMMAN75_R5 --first-day 0   --days 100 --model results/models_AMMAN75_R5_ugv1_fastest.pkl
    python run_wind.py C75_R5     --first-day 300 --days 100 --model results/models_C75_R5_ugv1_fastest.pkl
    python run_wind.py RC100_R5   --first-day 0   --days 100 --model results/models_RC100_R5_ugv1_fastest.pkl

The days are the held-out days of each learned model (the test days are the FIRST days in
`train_policy.py`; C75_R5 uses the fresh days 300+).  A first run on days 230-329 (Amman) and
150-249 (RC100_R5) used training days by mistake; it is kept as `results/wind_*_traindays.txt`
and is not reported.

| Setting | In-flight depletion, plans without wind | Fastest - regret | Learned - fastest | Ceiling - fastest |
| --- | --- | --- | --- | --- |
| Amman, 75 | 55% of days, 0.97/day | -0.1%, n.s. | -0.6% [-39.1, +6.0], n.s. | -1.7% [-70.4, -23.0] |
| C75_R5 | 74% of days, 1.76/day | -7.6% [-524, -234] | -1.7% [-161.1, +0.0], n.s. | -7.8% [-484, -241] |
| RC100_R5 | 91% of days, 3.31/day | -4.2% [-490, -272] | -3.0% [-362.1, -152.6] | -3.4% [-391, -202] |

Re-planned with the wind at the start of the day and a 10% reserve, in-flight depletion falls
to 0.01-0.09 a day.  The reserved model finds no feasible charging option 0.8 (C75_R5) and 4.4
(RC100_R5) times a day; the drone then continues and is penalised (same for all rules).

