# Objectives

Here are some candidate tactical objectives for a time horizon ending on `OBJECTIVE_HORIZON` ($=T^+$).

## Acceptance criteria: static

The simplest way to formulate objectives is via bounding an observation on the target horizon date. All such objectives have the form $X\in\mathcal{A}$ where $X$ is some observable and $\mathcal{A}$ is the (outcome) acceptance set.

* Yield $Y$ below target $\tau_Y$ on `OBJECTIVE_HORIZON`.
* Emissions $E$ below target $\tau_E$ on `OBJECTIVE_HORIZON`.
* Participation $P$ within target range $[\tau_P^-,\tau_P^+]$ on `OBJECTIVE_HORIZON`.

A smarter generalisation replaces the instantaneous observation with an aggregate over some longer time, say monthly average of daily observations.

## Acceptance criteria: dynamic

For some variables, we would also like to formulate *maintenance objectives* which involve regular measurements from the decision date until the objective horizon. The measurements are defined by some tick set $T_m\subset [T^-,T^+]$. Typically, the measurement tick set will be a subset of the data tick set.

After all measurements are taken, each acceptance criterion $X\in\mathcal{A}$ defines a *visit set* $T_{X\in\mathcal{A}}\subseteq T_m$ of ticks at which the measurements satisfied the criterion. Maintenance objectives are then defined by conditioning on the visit times $T_{X\in\mathcal{A}}$.

A basic criterion is
$$
\frac{|T_{X\in\mathcal{A}}|}{|T|} \geq p
$$
for some target $p\in[0,1]$.

## Objective choices

We define the following parameters (expressed in Python pseudocode):

```python
# measurement ticks
MEASUREMENT_INTERVAL = timedelta("1 month")
MEASUREMENT_START_DATE = date("2025-12-01")
MEASUREMENT_END_DATE = date("2026-07-01")

DATA_RESOLUTION_INTERVAL = timedelta("1 day")

MEASUREMENT_TICKS = range(
    MEASUREMENT_START_DATE, 
    MEASUREMENT_END_DATE + MEASUREMENT_INTERVAL, 
    MEASUREMENT_INTERVAL
) # T_m

# for now, ignore issues arising from months 
# with different numbers of days
```

Our measurement tick set $T_m$ therefore consists of midnight UTC on the first day of each of the eight months from December 2025 to July 2026 (inclusive).

*Maintenance* objectives are measured every `MEASUREMENT_INTERVAL` from the start date until the end date (inclusive).

*Aggregate* objectives are measured in terms of an accumulated cost or return over time.

What exactly is measured on each measurement tick? The resolution of our data is *daily*.

### Maintenance objectives

While the community has signaled that it would tolerate participation rates below the current protocol target of $50\%$, there are still some limitations on accepted ranges. 

Suppose given objective parameters $0\leq\tilde\tau_P \leq \tau_P \leq 1$. The parameters are to be interpreted as follows:
* $\tau_P$ is the *attention threshold.* If participation drops below $\tau_P$, the community should meet and decide whether action needs to be taken.
* $\tilde\tau_P$ is the *response threshold.* If participation drops below $\tilde\tau_P$, the community must respond urgently.


Here are the participation maintenance objectives:

* (PM1) — $|T_{P\in\mathcal{A}_P}| \geq 7$. That is, $P$ exceeds the attention threshold for at least 7 of the 8 months. This corresponds to a maintenance threshold of $p=7/8$.
* (PM2) — $|T_{P\in\tilde{\mathcal{A}}_P}| \geq 8$. That is, $P$ exceeds the response threshold for all eight months.

In Python pseudocode:

```python
OBJ_PM1 = len(visit_times(participation, ATTENTION_PARTICIPATION) >= 7
OBJ_PM2 = len(visit_times(participation, RESPOND_PARTICIPATION)) >= 8
```
where `visit_times(x,A)` is a function returning the set of measurement ticks for which `x in A`.


### Aggregate objectives

Community discussions have centered around the matter of managing emissions or yield. Common to both metrics is that they represent an accumulation over time.

If $I= T_m \cap [t_-,t_+]$ is a subinterval, we denote by $E(I)$ the *observed emissions over the interval $I$*. That is, 

$$
E(I) = \frac{S_{t_+} - S_{t_-}}{S_{t_+}}.
$$

We now introduce the aggregate objective:

* (ET) — $E(I) \leq \tau_E$. The corresponding acceptance set is $\mathcal{A}_{E(I)} = [0,\tau_E]$.

Similar objectives could be introduced for yield, but as the idea of making this an objective seems to remain controversial, we don't formulate this.

In Python pseudocode:
```python
OBJ_ET_INTERVAL = (OBJ_ET_START_DATE, OBJ_ET_END_DATE)
emission_measured = (total_supply[OBJ_ET_END_DATE] - total_supply[OBJ_ET_START_DATE]) / total_supply[OBJ_ET_END_DATE]
obj_et = emission_observed <= OBJ_ET_UPPER
```
