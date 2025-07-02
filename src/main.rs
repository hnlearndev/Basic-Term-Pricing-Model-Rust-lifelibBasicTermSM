mod assumptions;
mod mp;
mod projection;

use crate::assumptions::{
    AssumptionSet, get_acq_exp_df, get_inf_rate_df, get_lapse_df, get_load_rate_df, get_mort_df,
    get_mtn_exp_df, get_spot_rate_df,
};
use crate::mp::generate_model_points;
use crate::projection::project_model_points;
use std::time::Instant;

use polars::prelude::*;

fn main() -> Result<(), PolarsError> {
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

    let full_proj = project_model_points(&model_points_df, &assumption_set)?;

    println!("{:#?}", full_proj);

    let duration = start.elapsed();
    println!("Elapsed: {:.2?}", duration);

    Ok(())
}
