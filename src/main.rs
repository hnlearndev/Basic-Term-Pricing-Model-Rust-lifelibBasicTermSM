mod assumptions;
mod model_points;
mod projection;
mod queries;

use crate::assumptions::{
    AssumptionSet, get_acq_exp_df, get_inf_rate_df, get_lapse_df, get_load_rate_df, get_mort_df,
    get_mtn_exp_df, get_spot_rate_df,
};
use crate::model_points::generate_model_points;
use crate::projection::{project_model_points, project_model_points_parallel};
use crate::queries::{get_ave_prem_rate_by_age, get_avg_prem_by_age_and_term};
use polars::prelude::*;
use std::time::Instant;

fn main() -> Result<(), PolarsError> {
    // Start timer
    let start = Instant::now();

    let model_points_df = generate_model_points(10_000, 987654321)?;

    let assumption_set = AssumptionSet {
        mort: get_mort_df("cso80")?,
        lapse: get_lapse_df("lapse_01")?,
        inf: get_inf_rate_df("inf_01")?,
        acq: get_acq_exp_df("acq_exp_01")?,
        mtn: get_mtn_exp_df("mtn_exp_01")?,
        spot: get_spot_rate_df("spot_rate_01")?,
        load: get_load_rate_df("load_01")?,
    };

    let full_proj = project_model_points_parallel(&model_points_df, &assumption_set)?;

    // Print the first few rows of the projected DataFrame
    println!("Full projection {:#?}", full_proj);

    // Premium rates
    let prem_rate = get_avg_prem_by_age_and_term(&full_proj)?;
    println!("Average premium rate by age and term:\n{:#?}", prem_rate);

    // Log the elapsed time
    let duration = start.elapsed();
    println!("Elapsed: {:.2?}", duration);

    Ok(())
}
