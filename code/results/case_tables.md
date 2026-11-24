### Table A. Mean cost per day (late orders per day in brackets)

| Setting | Days | Regret | Nearest | Fastest | Learned | Ceiling |
| --- | --- | --- | --- | --- | --- | --- |
| AMMAN75_R5, 1 UGV, 50% dynamic | 100 | 2752.0 (0.09) | 2758.6 (0.06) | 2756.9 (0.07) | 2755.6 (0.05) | 2731.6 (0.01) |
| AMMAN150_R5, 1 UGV, 50% dynamic | 60 | 5014.4 (0.15) | 5020.8 (0.08) | 5030.0 (0.15) | 5022.9 (0.08) | 4992.4 (0.05) |

### Table B. Paired differences, percent of the reference rule's cost (95% CI of the difference in cost units; * = significant)

| Setting | Fastest vs regret | Fastest vs nearest | Learned vs fastest | Ceiling vs fastest | In-air rollout vs fastest | Decisions with a choice | Changed by ceiling |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AMMAN75_R5, 1 UGV, 50% dynamic | +0.2% [-1, +11] | -0.1% [-11, +7] | -0.0% [-10, +8] | -0.9%* [-39, -11] | -1.0%* [-42, -15] | 7.7 | 3.3 |
| AMMAN150_R5, 1 UGV, 50% dynamic | +0.3% [-5, +36] | +0.2% [-9, +28] | -0.1% [-30, +15] | -0.7%* [-66, -10] | -0.8%* [-72, -12] | 11.5 | 3.9 |

Counts over 2 settings: fastest significantly better than regret in 0, worse in 0; than nearest in 0, worse in 0; learned better than fastest in 0, worse in 0; ceiling better than fastest in 2; in-air rollout better than fastest in 2, worse in 0.
