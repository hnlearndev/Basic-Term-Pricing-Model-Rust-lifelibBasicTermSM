use crate::projection::RunResult;
use polars::prelude::*;

//---------------------------------------------------------------------------------------------------------
// PUBLIC
//---------------------------------------------------------------------------------------------------------
pub fn get_avg_prem_rate_by_age_and_term(run_result: &RunResult) -> PolarsResult<DataFrame> {
    let proj_df = run_result.projected_df.clone();

    // Compute rate column and aggregate
    let df = proj_df
        .lazy()
        .with_column((col("prem_pp") / col("sum_insured") * lit(1000.0)).alias("rate"))
        .filter(col("t").eq(lit(0)))
        .group_by([col("age"), col("term")])
        .agg([col("rate").mean().alias("avg_rate")])
        .sort(["term", "age"], Default::default())
        .collect()?;

    Ok(df)
}
