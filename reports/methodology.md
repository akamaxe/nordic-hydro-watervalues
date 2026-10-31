# Methodology

This document states the model in full. The notebooks show it at work on the data, and the code in `src/watervalues/` implements it; every symbol here has a counterpart in `config/base.yaml` or in the code.

Units throughout: energy in GWh, power in MW, prices in EUR/MWh and costs in kEUR (GWh × EUR/MWh = kEUR). Weeks are ISO weeks in UTC. The model year has 52 weekly stages: in years with an ISO week 53, that week is folded into week 52, and its energies are rescaled to a standard 168-hour stage.

## 1. Data

All series come from the ENTSO-E Transparency Platform for 2015–2025: Swedish generation per production type, total load (the country and the four bidding zones), day-ahead prices (SE1–SE4 and Germany) and the aggregated weekly filling of Swedish reservoirs. They are aggregated to weeks by time, never by counting rows, because the resolution changes from hourly to 15 minutes during the period.

The Swedish price is the load-weighted average of the four zones. The German price is DE-AT-LU until 30 September 2018 and DE-LU after.

The reservoir value published for week *w* is the storage at the end of that week. Two references verify this: Energiföretagen's weekly report, and its weekly statistics for 2019–2025. The physics of winter weeks also rules out the labels shifted by one week, though on its own it cannot rule out a weekly mean ([data/README.md](../data/README.md)). A one-week shift would pair the production of one week with the storage change of another and corrupt every inflow.

## 2. Inflow

With $S_w$ the storage at the start of week $w$, $P_w$ the hydro production and $Sp_w$ the spill, the energy balance of the reservoir gives the inflow

$$I_w = (S_{w+1} - S_w) + P_w + Sp_w.$$

- **Spill.** It is not observed and is taken as zero, so very wet weeks are underestimated.
- **Production.** $P_w$ covers all hydro, so $I_w$ is the total energy inflow.
- **Negatives.** A negative balance down to −150 GWh is storage-reporting noise. It is clipped to zero and listed, and anything below that stops the pipeline.
- **Gap weeks.** The two weeks of 2015 without production data get the mean inflow of the same week in the other years.

Each ISO year is one equally likely inflow scenario: 11 scenarios × 52 stages.

## 3. Price curve and cost of net import

Residual demand is what hydro and trade must cover, and net import is the part trade covers:

$$D_w = L_w - G^{nuc}_w - G^{wind}_w - G^{solar}_w - G^{other}_w, \qquad N_w = D_w - P_w.$$

The weekly Swedish price is explained by the German price, net import and a seasonal intercept for the calendar quarter:

$$p_w = \alpha + s_{q(w)} + \gamma\, p^{DE}_w + b_w N_w + \varepsilon_w, \qquad b_w = \kappa\, p^{DE}_w.$$

The slope proportional to the German price replaces a constant slope $b_w = \beta$. Fitted year by year, the constant slope varies by an order of magnitude and follows the German price level.

Eight variants were compared:
- the slope constant or proportional;
- with or without seasonal intercepts;
- with or without the crisis years 2021–2023.

The base model was chosen by the following rule:

1. The lowest leave-one-year-out RMSE of the weekly price, with each year predicted by a fit that has not seen it.
2. Variants within 0.5 EUR/MWh of that RMSE count as tied. The smallest out-of-sample seasonal bias decides between them, because the water value model moves water between seasons.
3. If still tied, the fewest parameters.

Coefficients are OLS estimates with Newey–West standard errors (13 lags). Weeks with a missing input are left out of the fit.

For a 52-week German price profile, every variant reduces to a weekly intercept $a_t = \alpha + s_{q(t)} + \gamma\, p^{DE}_t$ and slope $b_t$. The cost of covering a net import $N$ is the area under the price curve,

$$C_t(N) = a_t N + \tfrac12 b_t N^2 \;\; [\text{kEUR}], \qquad C_t'(N) = p_t(N).$$

It is negative for exports (revenue) and convex because $b_t > 0$, which the code checks in every stage.

## 4. Water values

The state is the stored energy $S$ at the start of a week. For week $t$ and inflow scenario $j$ (probability $\pi_j = 1/J$), the expected cost from now on is

$$V_t(S) = \sum_{j} \pi_j \min_{R,\,Sp} \Big[\, C_t(D_t - R) + c^{sp} Sp + \delta\, V_{t+1}(S') \,\Big],$$

$$S' = S + I_{t,j} - R - Sp, \qquad 0 \le S' \le S^{max}, \qquad \min(R^{min}_t, S + I_{t,j}) \le R \le R^{max}, \qquad Sp \ge 0.$$

| Symbol | Meaning | Value |
|---|---|---|
| $D_t$ | mean residual demand of the week over the years | from the data |
| $S^{max}$ | largest storage observed | from the data |
| $R^{max}$ | installed capacity × 168 h × availability | 16,300 MW (assumed; still to be checked against an official source) × 168 h × 0.85 |
| $R^{min}_t$ | 5th percentile over the years of min(production, inflow) of the week | from the data |
| $c^{sp}$ | spill penalty | 0.01 EUR/MWh |
| $\delta$ | weekly discount factor | $1.05^{-1/52}$ |

Inflow is known when the week's decision is made (hazard-decision) and independent between weeks.

**The minimum release.** Forced production cannot exceed what flows in during the week, so $R^{min}_t$ is a low percentile of min(production, inflow), not of production alone. A floor on production would treat output that follows demand, as in winter, as a physical minimum. At low storage it would force water out at the week's price and make the water value rise with storage. The inflow-capped floor removes most of that effect. A small rise remains at a nearly empty reservoir, from the kink of the rule itself.

**Release and spill.** For the water let out, $X = R + Sp$, the cost is convex in $R$. It is smallest where the price falls to $-c^{sp}$, so the release stops there and the rest is spilled. While prices are positive this means spilling only above turbine capacity. The linear curve reaches negative prices before full release in many weeks, because it is extrapolated beyond the largest observed export.

**The German price profile.** The base case takes the seasonal shape of the years outside 2021–2023, scaled to the mean of all years. Scenarios rescale the same shape to other annual means.

The water value is $w_t(S) = -\partial V_t / \partial S$ [EUR/MWh]. Where no bound binds, the optimality condition is

$$p_t = a_t + b_t (D_t - R^*) = \delta\, w_{t+1}(S'),$$

which states that the price of the week equals the discounted water value of the next week.

**Solution method.**
- $K = 201$ uniform storage points.
- Backward over the 52 weeks, with the next storage on the same grid, vectorised over (storage, scenario, next storage). Only the loop over weeks remains: on a uniform grid the immediate cost depends only on $k - k'$ and the scenario.
- The year is repeated with the value of week 1 as the end value until no water value changes by more than 0.05 EUR/MWh between cycles. The level of $V$ keeps drifting after that; only its differences are used.

**Checks.**
- In the base case and the price scenarios, water values are non-increasing in storage within the price resolution of the grid. In most backtest years they rise slightly at a nearly empty reservoir, at the kink of the minimum release; the simulated reservoir never gets that low (`outputs/simulate_report.txt` lists every case).
- No water value is below $-c^{sp}$.
- The optimality condition holds at every interior decision, within half a grid step of the one-sided water values.
- With 101 or 401 storage points instead of 201, the water values change by a small fraction of a EUR/MWh on average. The largest differences sit near an empty or a full reservoir (notebook 04).

## 5. Simulation, backtest and scenarios

**Simulation.** Week by week, the release is chosen on a 10 GWh grid to minimise $C_t(D_t - R) + c^{sp} Sp + \delta\, V_{t+1}(S')$, with $V_{t+1}$ interpolated linearly in storage. Water is spilled only when the reservoir would overflow; since no water value is below $-c^{sp}$, spilling a reservoir that is not full is never strictly better. The simulated price is $p_t = a_t + b_t (D_t - R_t)$.

**Backtest of year $y$.**
- Start from the observed storage, with the observed inflow, residual demand and German price.
- Solve the water values with that year's residual demand and German price. This gives perfect foresight of both, so the remaining uncertainty is hydrological.
- Refit the price curve without year $y$, and leave $y$ out of the inflow scenarios. The minimum release and $S^{max}$ still come from all years; that effect was checked to be negligible.
- A week without an observed residual demand uses the mean of that week over the years, and a missing German price takes the base seasonal shape at the year's own level. These weeks, and the weeks with an imputed inflow, are left out of the price metrics. The folded week 52 of a 53-week year is left out of the storage error, because its observed end of week is a week later.
- Metrics: the weekly price correlation, MAE and bias, and the RMSE of storage at the end of each week, in % of $S^{max}$.

**Scenarios.**
- The German price profile scaled to annual means of 50, 80 and 120 EUR/MWh, with the water value read along the mean observed storage path.
- The driest and the wettest inflow year, simulated from the same initial storage with the base-case water values.

## 6. Limitations

- **One reservoir, no grid.** Sweden is a single aggregated reservoir, with no transmission limits between the northern and southern zones.
- **Inflow.** It is independent between weeks, there are only eleven historical years, and spill is not observed.
- **The price curve.** It is linear, reduced-form and estimated by OLS. Hydro responds to the price, so net import is not exogenous. It has no scarcity premium at low storage and is extrapolated at large exports.
- **Fixed demand and German price.** Residual demand and the German price are deterministic inside the water value model.
- **Risk neutrality.** The model minimises expected cost.
- **Foresight in the backtest.** It knows each year's residual demand and German price in advance.
