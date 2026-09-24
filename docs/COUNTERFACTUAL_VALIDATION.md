# Counterfactual validation plan

BallDNA separates prediction of observed team profiles from prediction of roster
transactions. A model can score real team-seasons accurately while learning
correlations that fail when a player is added or removed. The two tasks therefore
need different tests.

## Validation layers already implemented

1. **Observed team-profile validation** — leave-one-season-out testing reports Net
   Rating MAE, RMSE, and rank correlation for complete historical team-seasons.
2. **Historical-support detection** — a simulated roster is compared with the
   nearest historical team in standardized embedding space. The 95th percentile
   of leave-one-out historical distances is the support boundary.
3. **Rotation invariants** — automated tests require exactly 240 team minutes,
   position-aware displacement, source MPG as a ceiling, click-order-independent
   allocation, auditable minute transfers, reversible forced-role experiments,
   and position-aware frontcourt charts.
4. **Monotonic upside behavior** — the experimental talent lens uses non-negative
   coefficients for rotation and top-three quality. Increasing either input cannot
   reduce its score.

These checks validate software behavior and observed-profile prediction. They do
not yet establish that a displayed roster change predicts a real transaction.

## Next empirical backtest

Build a season-to-season evaluation table in which every feature is available
before the target season:

1. Use season `t` player vectors and roles as inputs.
2. Identify players retained, added, and removed before season `t+1`.
3. Construct the new roster without using `t+1` performance or minutes.
4. Predict the change from team `t` to team `t+1`.
5. Evaluate Net Rating change MAE, win-change MAE, directional accuracy, rank
   correlation, interval coverage, and error by move size.
6. Compare against simple baselines: no change, prior record, and sum of prior
   player quality.
7. Report results separately for stars, rotation players, rookies, high roster
   turnover, and out-of-distribution teams.

The split must be chronological. Hyperparameters should be selected only on
seasons before the test season. Rookies and players missing prior-season data need
an explicit replacement-level prior rather than leaked current-season statistics.

## Higher-quality model after the backtest

The next model should allocate roles jointly under a 240-minute constraint and a
team possession/usage budget. Lineup possessions, on/off or regularized adjusted
plus-minus features, availability, age, and interactions between creation,
spacing, defense, and position would help model diminishing returns. Predictions
should include calibrated intervals and abstain when historical support is weak.

Until that backtest exists, the UI calls the monotonic output **experimental
expected wins**: a useful roster-ranking point estimate, not a calibrated causal
trade forecast.
