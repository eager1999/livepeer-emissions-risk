# Objectives

Here are some candidate tactical objectives for a time horizon ending on `OBJECTIVE_HORIZON` ($=T^+$).

First, the simplest way to formulate objectives is via bounding an observation on the target horizon date. All such objectives have the form $X\in\mathcal{A}$ where $X$ is some observable and $\mathcal{A}$ is the (outcome) acceptance set.

* Yield $Y$ below target $\tau_Y$ on `OBJECTIVE_HORIZON`.
* Emissions $E$ below target $\tau_E$ on `OBJECTIVE_HORIZON`.
* Participation $P$ within target range $[\tau_P^-,\tau_P^+]$ on `OBJECTIVE_HORIZON`.

A smarter generalisation replaces the instantaneous observation with an average over some longer time, say monthly average of daily observations.

For some variables, we would also like to formulate *maintenance objectives* which involve regular measurements from the decision date until the objective horizon. The measurements are defined by some tick set $T\subset [T^-,T^+]$. 

After all measurements are taken, each acceptance criterion $X\in\mathcal{A}$ defines a subset $T_{X\in\mathcal{A}}\subseteq T$ of ticks at which the measurements satisfied the criterion. Maintenance objectives are then defined by conditioning on the satisfaction set $T_{X\in\mathcal{A}}$.

A basic criterion is
$$
\frac{\#T_{X\in\mathcal{A}}}{\#T} \geq p
$$
for some target $p\in[0,1]$.

### Objective choices

We define the following parameters (expressed in Python pseudocode):

```python
# measurement ticks
MEASUREMENT_INTERVAL = timedelta("1 month")
MEASUREMENT_START_DATE = date("2025-12-01")
MEASUREMENT_END_DATE = date("2026-07-01")

DATA_RESOLUTION_INTERVAL = timedelta("1 day")

# for now, ignore issues arising from months 
# with different numbers of days
```

*Maintenance* objectives are measured every `MEASUREMENT_INTERVAL` from the start date until the end date (inclusive).

*Goalpost* objectives are measured only on the end date.

What exactly is measured on each measurement tick? The resolution of our data is *daily*.