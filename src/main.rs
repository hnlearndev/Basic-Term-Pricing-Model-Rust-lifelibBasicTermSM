mod assumptions;
mod model_points;
mod prem_rate;
mod projection;

use crate::assumptions::{
    AssumptionSet, get_acq_exp_df, get_inf_rate_df, get_lapse_df, get_load_rate_df, get_mort_df,
    get_mtn_exp_df, get_spot_rate_df,
};
use crate::model_points::generate_model_points;
use crate::prem_rate::get_avg_prem_rate_by_age_and_term;
use crate::projection::{
    RunSetup, project_multiple_run_parallel, project_multiple_run_parallel_x2,
};
use polars::prelude::*;
use std::time::Instant;

//---------------------------------------------------------------------------------------------------------
// PRIVATE
//---------------------------------------------------------------------------------------------------------
fn get_run_setups() -> PolarsResult<Vec<RunSetup>> {
    let model_points_df = generate_model_points(10_000, 987654321)?;

    // Let construct muttiple runs with different assumptions
    let assumption_set_01 = AssumptionSet {
        mort: get_mort_df("cso80")?,
        lapse: get_lapse_df("lapse_01")?,
        inf: get_inf_rate_df("inf_01")?,
        acq: get_acq_exp_df("acq_exp_01")?,
        mtn: get_mtn_exp_df("mtn_exp_01")?,
        spot: get_spot_rate_df("spot_rate_01")?,
        load: get_load_rate_df("load_01")?,
    };

    let assumption_set_02 = AssumptionSet {
        mort: get_mort_df("cso80")?,
        lapse: get_lapse_df("lapse_02")?,
        inf: get_inf_rate_df("inf_02")?,
        acq: get_acq_exp_df("acq_exp_02")?,
        mtn: get_mtn_exp_df("mtn_exp_02")?,
        spot: get_spot_rate_df("spot_rate_01")?,
        load: get_load_rate_df("load_02")?,
    };

    let run_setup_01 = RunSetup {
        description: "Run setup 01".to_string(),
        model_points_df: model_points_df.clone(),
        assumptions: assumption_set_01,
    };

    let run_setup_02 = RunSetup {
        description: "Run setup 02".to_string(),
        model_points_df: model_points_df.clone(),
        assumptions: assumption_set_02,
    };

    Ok(vec![run_setup_01, run_setup_02])
}

//---------------------------------------------------------------------------------------------------------
// MAIN
//---------------------------------------------------------------------------------------------------------
fn main() -> Result<(), PolarsError> {
    // Start timer
    let start = Instant::now();

    let run_setups = get_run_setups()?;

    let multi_run_results = project_multiple_run_parallel_x2(&run_setups)?;

    // Print aggregated results of multiple runs
    let multi_run_proj_df = multi_run_results.aggregate_projected_df()?;
    println!("Full projection {:#?}", multi_run_proj_df);

    // Premium rates - assume that run 01 is used to obtain premium rates
    let run_result_01 = &multi_run_results.run_results[0];
    let prem_rate_df = get_avg_prem_rate_by_age_and_term(run_result_01)?;
    println!(
        "Average premium rate by age and term under setup 01:\n{:#?}",
        prem_rate_df
    );

    // Log the elapsed time
    let duration = start.elapsed();
    println!("Elapsed: {:.2?}", duration);

    Ok(())
}
