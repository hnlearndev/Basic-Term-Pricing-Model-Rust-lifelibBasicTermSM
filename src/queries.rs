use polars::prelude::*;

pub fn get_ave_prem_rate_by_age(proj_df: &DataFrame, term: i32) -> PolarsResult<DataFrame> {
    // Filter for the given term and t, then compute average (net_prem/sum_insured*1000) by age
    let df = proj_df
        .clone()
        .lazy()
        .filter(col("term").eq(lit(term)).and(col("t").eq(lit(0))))
        .with_column((col("prem_pp") / col("sum_insured") * lit(1000.0)).alias("rate"))
        .group_by([col("age")])
        .agg([col("rate").mean().alias("avg_rate")])
        .sort(["age"], Default::default())
        .collect()?;
    Ok(df)
}

pub fn get_ave_prem_rate_by_age_and_term(
    proj_df: &DataFrame,
    term: i32,
) -> PolarsResult<DataFrame> {
    // Filter for the given term and t, then compute average (net_prem/sum_insured*1000) by age
    let df = proj_df
        .clone()
        .lazy()
        .filter(col("term").eq(lit(term)).and(col("t").eq(lit(0)))) // Get the term and first premium_pp
        .with_column((col("prem_pp") / col("sum_insured") * lit(1000.0)).alias("rate"))
        .group_by([col("age")])
        .agg([col("rate").mean().alias("avg_rate")])
        .sort(["age"], Default::default())
        .collect()?;
    Ok(df)
}

pub fn get_avg_prem_by_age_and_term(proj_df: &DataFrame) -> PolarsResult<DataFrame> {
    // Compute rate column and aggregate
    let df = proj_df
        .clone()
        .lazy()
        .with_column((col("prem_pp") / col("sum_insured") * lit(1000.0)).alias("rate"))
        .filter(col("t").eq(lit(0)))
        .group_by([col("age"), col("term")])
        .agg([col("rate").mean().alias("avg_rate")])
        .sort(["term", "age"], Default::default())
        .collect()?;

    // Use DataFrame::pivot_stable for pivoting (Polars >=0.29)
    Ok(df)
}
