import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import altair as alt
    import os, json
    return alt, json, mo, os, pl


@app.cell
def _():
    # Rounds calculations
    SECONDS_PER_ETH_SLOT = 12
    ETH_SLOTS_PER_ROUND = 6337
    ROUNDS_PER_YEAR = (3600 * 24 * 365) / (SECONDS_PER_ETH_SLOT * ETH_SLOTS_PER_ROUND)
    return (ROUNDS_PER_YEAR,)


@app.cell
def _(ROUNDS_PER_YEAR, json, os, pl):
    # Data loading
    DATA_PATH = os.getenv("LPT_DATA_PATH", "../data/lpt-daily-data-22-24.json")

    with open(DATA_PATH) as h:
        data_json = json.load(h)

    # Convert JSON data to DataFrame, coercing "total-supply" and "bonded" columns to float64 and "date" to DateTime objects
    def load_df_from_json(json):
        return pl.DataFrame(data_json).with_columns([
            pl.col("date").str.strptime(pl.Date, format="%Y-%m-%d"),
            pl.col("total-supply").cast(pl.Float64),
            pl.col("bonded").cast(pl.Float64)
        ])

    def prepare_bonding_metrics(df):
        return df.with_columns([
            (pl.col("bonded") / pl.col("total-supply")).alias("bonding-rate"),
            (pl.col("bonded") / (pl.col("total-supply") - pl.col("bonded"))).alias("bonding-ratio")
        ]).with_columns([
            (pl.col("bonding-rate") * 100).alias("bonding-rate-%"),
            (pl.col("bonding-rate").diff() * 100 ).alias("bonding-rate-change-%")
        ])

    def prepare_emission_metrics(df):
        return df.with_columns([
            (pl.col("inflation") / 1_000_000_000).alias("emission-rate-round"),
            (((1 + (pl.col("inflation") / 1_000_000_000)) ** ROUNDS_PER_YEAR) - 1).alias("emission-rate-annualised")
        ]).with_columns([
            (1 - 1 / (1 + pl.col("emission-rate-round"))).alias("dilution-rate-round"),
            (1 - 1 / (1 + pl.col("emission-rate-annualised"))).alias("dilution-rate-annualised")
        ]).with_columns([
            (pl.col("dilution-rate-round") * 100).alias("dilution-rate-round-%"),
            (pl.col("dilution-rate-annualised") * 100).alias("dilution-rate-annualised-%")
        ])

    def prepare_yield_metrics(df):
        return df.with_columns([
            (pl.col("emission-rate-round") / pl.col("bonding-rate")).alias("yield-round"),
            (pl.col("emission-rate-annualised") / pl.col("bonding-rate")).alias("yield-annualised")
        ]).with_columns([
            ((1 + pl.col("yield-annualised")) * (1 - pl.col("dilution-rate-annualised")) - 1).alias("adjusted-yield-annualised"),
            (1 + pl.col("yield-round")).cum_prod().alias("yield-cumulative")
        ]).with_columns([
            (pl.col("yield-annualised") * 100).alias("yield-annualised-%"),
            (pl.col("adjusted-yield-annualised") * 100).alias("adjusted-yield-annualised-%")
        ])

    def prepare_adjusted_bonding_rate(df):
        return df.with_columns([
            (pl.col("bonded") / pl.col("yield-cumulative")).alias("adjusted-bonded")
        ]).with_columns([
            (100 * pl.col("adjusted-bonded") / df["total-supply"][0]).alias("adjusted-bonding-rate-%")
        ])

    # Trailing yields
    SHIFT = int(ROUNDS_PER_YEAR)

    def prepare_trailing_returns(df):
        return df.with_columns([
            (pl.col("yield-cumulative") / pl.col("yield-cumulative").shift(SHIFT)).alias("return-trailing-1y"),
            (pl.col("total-supply").shift(SHIFT) / pl.col("total-supply")).alias("dilution-trailing-1y")
        ]).with_columns([
            (pl.col("return-trailing-1y") * pl.col("dilution-trailing-1y")).alias("adjusted-return-trailing-1y")
        ]).with_columns([
            ((pl.col("return-trailing-1y") - 1) * 100).alias("return-trailing-1y-%"),
            ((pl.col("adjusted-return-trailing-1y") - 1) * 100).alias("adjusted-return-trailing-1y-%")
        ]).filter(pl.col("return-trailing-1y-%").is_not_null())

    PREPARATIONS = [
        load_df_from_json, 
        prepare_bonding_metrics, 
        prepare_emission_metrics, 
        prepare_yield_metrics, 
        prepare_adjusted_bonding_rate, 
        prepare_trailing_returns
    ]
    return PREPARATIONS, data_json


@app.cell
def _(PREPARATIONS, data_json):
    from functools import reduce
    df = reduce(lambda dfa, dfb: dfb(dfa), PREPARATIONS, data_json)
    df
    return (df,)


@app.cell
def _(alt, df):
    dilution_rate_chart = alt.Chart(df).mark_line(color="magenta").encode(
        x="date:T",
        y=alt.Y("dilution-rate-annualised-%:Q", axis=alt.Axis(title="Annualised dilution rate (%)"))
    )

    dilution_rate_chart
    return


@app.cell
def _(alt, df, pl):
    df_yield_flattened = df.with_columns([
        pl.col("yield-annualised-%").alias("Nominal"),
        pl.col("adjusted-yield-annualised-%").alias("Adjusted")
    ]).unpivot(
        index="date", 
        on=["Nominal", "Adjusted"], 
        variable_name="metric", 
        value_name="value"
    )

    alt.Chart(df_yield_flattened).mark_line().encode(
        x="date:T",
        y=alt.Y("value:Q", axis=alt.Axis(title="Yield (%)")),
        color="metric:N",
        tooltip=["date:T", "value:Q", "metric:N"]
    )
    return


@app.cell
def _(alt, df):
    bonding_rate_chart = alt.Chart(df).mark_line(color="green").encode(
        x="date:T",
        y="bonding-rate-%:Q"
    )

    bonding_rate_index_chart = alt.Chart(df).mark_line(color="blue").encode(
        x="date:T",
        y="adjusted-bonding-rate-%:Q"
    )

    bonding_rate_chart
    return


@app.cell
def _(alt, df):
    bonding_ratio_chart = alt.Chart(df).mark_line(color="purple").encode(
        x="date:T",
        y="bonding-ratio:Q"
    )
    bonding_ratio_chart
    return


@app.cell
def _(alt, df):
    alt.Chart(df).mark_line(color="orange").encode(
        x="date:T",
        y=alt.Y("bonding-rate-change-%:Q", title="Bonding rate change (%)")
    )
    return


@app.cell
def _(df, mo, pl):
    min_inflation_offset = df.select(pl.col("inflation").arg_min()).item()

    mo.md(f"""
    **Date of minimum emissions:** {df["date"][min_inflation_offset].strftime("%Y-%m-%d")}\n
    **Minimum annualised dilution rate:** {df.select(pl.col("dilution-rate-annualised-%").min()).item():.2f}%\n
    **Minimum annualised yield:** {df.select(pl.col("yield-annualised-%").min()).item():.2f}%\n
    **Minimum annualised adjusted yield:** {df.select(pl.col("adjusted-yield-annualised-%").min()).item():.2f}%
    """)
    return


@app.cell
def _(alt, df, pl):
    df_trailing_flattened = df.with_columns([
        pl.col("return-trailing-1y-%").alias("Nominal"),
        pl.col("adjusted-return-trailing-1y-%").alias("Adjusted")
    ]).unpivot(
        index="date", 
        on=["Nominal", "Adjusted"], 
        variable_name="metric", 
        value_name="value"
    )

    alt.Chart(df_trailing_flattened).mark_line().encode(
        x="date:T",
        y=alt.Y("value:Q", axis=alt.Axis(title="Trailing 1Y returns (%)")),
        color="metric:N",
        tooltip=["date:T", "value:Q", "metric:N"]
    )
    return


@app.cell
def _(alt, df):
    import numpy as np

    shuft = 1

    df_diddle = df.with_columns([
        (np.log(df["bonding-ratio"] / df["bonding-ratio"].shift(shuft)) / shuft).alias("log-bonding-ratio-diff")
    ])

    alt.Chart(df_diddle).transform_density(
        "log-bonding-ratio-diff",
        as_=["log-bonding-ratio-diff", "density"]
    ).mark_area(color="lightblue", opacity=0.7).encode(
        x=alt.X("log-bonding-ratio-diff:Q", axis=alt.Axis(title="Log Bonding Ratio Diff")),
        y=alt.Y("density:Q", axis=alt.Axis(title="Density"))
    )
    return df_diddle, np


@app.cell
def _(df_diddle, mo, np, pl):
    diffs = df_diddle["log-bonding-ratio-diff"].drop_nulls()

    def sim_random_walk(runs = 100):
        outcomes = []
        for _ in range(runs):
            outcomes.append(diffs.sample(30, with_replacement=True).sum())
        return outcomes

    simulated_180d = pl.Series(sim_random_walk(10000))
    p05_180d = simulated_180d.quantile(0.975)


    def logit_to_percent(logit):
        ratio = np.exp(logit)
        return 100 * ratio / (1 + ratio)

    mo.md(f"""
    *Lower 95% confidence interval for 180-day log bonding ratio diff:* {logit_to_percent(p05_180d):.6f}
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
