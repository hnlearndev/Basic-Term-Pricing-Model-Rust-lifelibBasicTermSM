use polars::prelude::*;
use std::path::PathBuf;

//---------------------------------------------------------------------------------------------------------
// STRUCTS
//---------------------------------------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct AssumptionSet {
    pub mort: DataFrame,
    pub lapse: DataFrame,
    pub inf: DataFrame,
    pub acq: DataFrame,
    pub mtn: DataFrame,
    pub spot: DataFrame,
    pub load: DataFrame,
}

//---------------------------------------------------------------------------------------------------------
// PRIVATE
//---------------------------------------------------------------------------------------------------------
fn _get_assumption_df(
    file_path_str: &str,
    col_name: &str,
    new_col_name: &str,
) -> PolarsResult<DataFrame> {
    let lapse_file_path = PathBuf::from(file_path_str);

    CsvReadOptions::default()
        .try_into_reader_with_file_path(Some(lapse_file_path))?
        .finish()?
        .lazy()
        .select([col("year"), col(col_name).alias(new_col_name)]) // Select which column as a rate
        .collect()
}

//---------------------------------------------------------------------------------------------------------
// PUBLIC
//---------------------------------------------------------------------------------------------------------
// Lapse assumption
pub fn get_lapse_df(lapse_type: &str) -> PolarsResult<DataFrame> {
    _get_assumption_df("src/assumptions/lapse.csv", lapse_type, "lapse_rate")
}

// Inflation assumption
pub fn get_inf_rate_df(inf_type: &str) -> PolarsResult<DataFrame> {
    _get_assumption_df("src/assumptions/inflation.csv", inf_type, "inf_rate")
}

// Acquisition assumption
pub fn get_acq_exp_df(acq_exp_type: &str) -> PolarsResult<DataFrame> {
    _get_assumption_df(
        "src/assumptions/acq_exp.csv",
        acq_exp_type,
        "real_acq_exp_pp",
    )
}

// Maintenance assumption
pub fn get_mtn_exp_df(mtn_exp_type: &str) -> PolarsResult<DataFrame> {
    _get_assumption_df(
        "src/assumptions/mtn_exp.csv",
        mtn_exp_type,
        "real_mtn_exp_pp",
    )
}

pub fn get_spot_rate_df(spot_rate_type: &str) -> PolarsResult<DataFrame> {
    _get_assumption_df("src/assumptions/spot_rate.csv", spot_rate_type, "spot_rate")
}

pub fn get_load_rate_df(spot_rate_type: &str) -> PolarsResult<DataFrame> {
    _get_assumption_df("src/assumptions/load.csv", spot_rate_type, "load_rate")
}

// Mortality assumption: The schema is slightly different from other since it is based on gender
pub fn get_mort_df(mort_type: &str) -> PolarsResult<DataFrame> {
    let mort_file_path = PathBuf::from("src/assumptions/mort.csv");

    CsvReadOptions::default()
        .try_into_reader_with_file_path(Some(mort_file_path))?
        .finish()?
        .select([
            "age",
            &format!("{}_m", mort_type),
            &format!("{}_f", mort_type),
        ])
}

//---------------------------------------------------------------------------------------------------------
// UNIT TESTS
//---------------------------------------------------------------------------------------------------------
