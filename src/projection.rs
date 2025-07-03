use crate::assumptions::AssumptionSet;
use crate::model_points::{ModelPoint, convert_model_points_df_to_vector};
use ndarray::prelude::*;
use polars::prelude::*;
use rayon::prelude::*;

// Process data in chunks to avoid stack overflow
const CHUNK_SIZE: usize = 100;

//---------------------------------------------------------------------------------------------------------
// STRUCTS
//---------------------------------------------------------------------------------------------------------
#[derive(Clone, Debug)]
pub struct RunSetup {
    pub description: String, // Optional description for the run
    pub model_points_df: DataFrame,
    pub assumptions: AssumptionSet,
}

#[derive(Clone, Debug)]
pub struct RunResult {
    pub run_setup: RunSetup,
    pub projected_df: DataFrame,
}

#[derive(Clone, Debug)]
pub struct MultipleRunResult {
    pub run_setups: Vec<RunSetup>,
    pub run_results: Vec<RunResult>,
}

impl MultipleRunResult {
    pub fn aggregate_projected_df(&self) -> PolarsResult<DataFrame> {
        let mut lfs: Vec<LazyFrame> = Vec::with_capacity(self.run_results.len());

        for (i, run_result) in self.run_results.iter().enumerate() {
            let run_id = i as i32;
            let run_description = run_result.run_setup.description.clone();

            let lf = run_result.projected_df.clone().lazy().with_columns(vec![
                lit(run_id).alias("run_id"),
                lit(run_description).alias("run_description"),
            ]);
            lfs.push(lf);
        }

        let lf = concat(lfs, Default::default())?;
        lf.collect()
    }
}

//---------------------------------------------------------------------------------------------------------
// PRIVATE
//---------------------------------------------------------------------------------------------------------
// Initialize projection lazyframe
fn __initialize_lf(
    id: i32,
    term: i32,
    entry_age: i32,
    sum_insured: f64,
) -> PolarsResult<LazyFrame> {
    let length = (term * 12 + 1) as usize; // Total months in the term

    let lf = df![
        "id" => vec![id; length],
        "term" => vec![term; length],
        "sum_insured" => vec![sum_insured; length],
        "t" => (0..= (length -1) as i32).collect::<Vec<i32>>(),
    ]
    .unwrap()
    .lazy()
    .with_column((col("t") / lit(12)).alias("duration"))
    .with_column((lit(entry_age) + col("duration")).alias("age"));

    Ok(lf.collect()?.lazy()) // To avoid nested lazyframe
}

// Map mortality assumption
fn __map_mort_assumption(
    lf: LazyFrame,
    mort_df: &DataFrame,
    gender: &str,
) -> PolarsResult<LazyFrame> {
    // Find the column name according to gender
    let suffix = format!("_{}", gender.to_lowercase());
    let mort_col_name = mort_df
        .get_column_names()
        .iter()
        .find(|&col| col.ends_with(&suffix))
        .ok_or_else(|| {
            polars::prelude::PolarsError::ComputeError(
                format!("Mortality column with suffix '{}' not found", suffix).into(),
            )
        })?
        .as_str();

    let mort_lf = mort_df
        .clone()
        .lazy()
        .select([col("age"), col(mort_col_name).alias("mort_rate")]);

    let result = lf
        .left_join(mort_lf, col("age"), col("age"))
        .with_column(col("mort_rate").fill_null(lit(0.0)).alias("mort_rate"));

    Ok(result.collect()?.lazy()) // To avoid nested lazyframe
}

// Map lapse, Inflation, Expenses and Spot rate assumption
fn __map_other_assumption(lf: LazyFrame, lookup_df: &DataFrame) -> PolarsResult<LazyFrame> {
    let col_name = lookup_df.get_column_names()[1].as_str();

    let lookup_lf = lookup_df
        .clone()
        .lazy()
        .with_column((col("year") - lit(1)).alias("duration")) // Adjust year to duration
        .select([col("duration"), col(col_name)]); // Drop "year" column

    let result = lf
        .left_join(lookup_lf, col("duration"), col("duration"))
        .with_column(col(col_name).fill_null(lit(0.0)).alias(col_name)); // Fill null with 0.0

    Ok(result.collect()?.lazy()) // To avoid nested lazyframe
}

// Discount factor from spot rate
fn __discount_factor(lf: LazyFrame) -> PolarsResult<LazyFrame> {
    let result = lf
        .with_column(
            // Spot rate monthly
            ((lit(1.0) + col("spot_rate")).pow(1.0 / 12.0) - lit(1.0)).alias("spot_rate_mth"),
        )
        .with_column(
            // Discount factor
            (lit(1.0) / (lit(1.0) + col("spot_rate_mth")).pow(col("t").cast(DataType::Float64)))
                .alias("discount_factor"),
        );

    Ok(result.collect()?.lazy())
}

// Expense per policy
fn __exp_pp(lf: LazyFrame) -> PolarsResult<LazyFrame> {
    let lf = lf
        .with_columns(vec![
            // Total real expense per policy
            (col("real_acq_exp_pp") + col("real_mtn_exp_pp")).alias("real_exp_pp"),
            // Inflation factor - for flat curve only
            (lit(1.0) + col("inf_rate"))
                .pow(col("t").cast(DataType::Float64) / lit(12.0))
                .alias("inf_factor"),
        ])
        .with_column(
            // Adjusted expense per policy
            (col("real_exp_pp") * col("inf_factor")).alias("exp_pp"),
        );

    Ok(lf.collect()?.lazy()) // To avoid nested lazyframe
}

// Count policy inforce and decrements
fn __policies_count(lf: LazyFrame, policy_count: f64, term: i32) -> PolarsResult<LazyFrame> {
    let lf = lf.with_columns(vec![
        // Monthly decrement rate
        (lit(1.0) - (lit(1.0) - col("mort_rate")).pow(1.0 / 12.0)).alias("mort_rate_mth"),
        (lit(1.0) - (lit(1.0) - col("lapse_rate")).pow(1.0 / 12.0)).alias("lapse_rate_mth"),
    ]);

    let df = lf.clone().collect()?;

    // Height of the dataframe
    let height = df.height() as usize;

    // Monthly mortality and lapse rate
    let mort_rate_mth = df.column("mort_rate_mth")?.f64()?.to_vec();
    let lapse_rate_mth = df.column("lapse_rate_mth")?.f64()?.to_vec();

    // Create a vector of 0.0 with length equal to lf.height()
    let mut pols_if = Array1::<f64>::zeros(height).to_vec();
    pols_if[0] = policy_count; // Set first element to policy_count

    let mut pols_maturity = Array1::<f64>::zeros(height).to_vec();
    let mut pols_death = Array1::<f64>::zeros(height).to_vec();
    let mut pols_lapse = Array1::<f64>::zeros(height).to_vec();

    for i in 0..(height - 1) {
        if i == 0 {
            pols_if[i] = policy_count;
        } else {
            pols_if[i] =
                pols_if[i - 1] - pols_maturity[i - 1] - pols_death[i - 1] - pols_lapse[i - 1];
        }

        pols_maturity[i] = if i == (term * 12) as usize {
            pols_if[i] // Maturity at the end of the term
        } else {
            0.0 // No maturity before term ends
        };

        pols_death[i] = (pols_if[i] - pols_maturity[i]) * mort_rate_mth[i].unwrap_or(0.0);
        pols_lapse[i] =
            (pols_if[i] - pols_maturity[i] - pols_death[i]) * lapse_rate_mth[i].unwrap_or(0.0);
    }

    // Create a DataFrame from these vectors
    let new_df = df![
        "pols_if" => pols_if,
        "pols_maturity" => pols_maturity,
        "pols_death" => pols_death,
        "pols_lapse" => pols_lapse,
    ]?;

    // Horizontally concatenate the new columns to the existing LazyFrame
    let result = df
        .hstack(&[
            new_df.column("pols_if")?.clone(),
            new_df.column("pols_maturity")?.clone(),
            new_df.column("pols_death")?.clone(),
            new_df.column("pols_lapse")?.clone(),
        ])?
        .lazy();

    Ok(result.collect()?.lazy()) // To avoid nested lazyframe
}

// Net premium calculation - do not consume dataframe
fn ___calculate_net_premium(lf: LazyFrame, sum_insured: f64) -> PolarsResult<f64> {
    // Calculate both PV claims and premium annuities in a single operation
    let result = lf
        .with_columns(vec![
            // Claim per policy
            lit(sum_insured).alias("claim_pp"),
            // Portfolio claims
            (lit(sum_insured) * col("pols_death")).alias("claims"),
        ])
        .with_columns(vec![
            // PV of claims and PV of premium annuities
            (col("claims") * col("discount_factor")).alias("pv_claims_component"),
            (col("pols_if") * col("discount_factor")).alias("pv_annuities_component"),
        ])
        .select([
            col("pv_claims_component").sum().alias("pv_claims"),
            col("pv_annuities_component").sum().alias("prem_annuities"),
        ])
        .collect()?;

    // Extract both values in one operation
    let pv_claims = result
        .column("pv_claims")?
        .get(0)?
        .extract::<f64>()
        .unwrap_or(0.0);

    let prem_annuities = result
        .column("prem_annuities")?
        .get(0)?
        .extract::<f64>()
        .unwrap_or(0.0);

    // Calculate net premium
    let net_premium = if prem_annuities != 0.0 {
        pv_claims / prem_annuities
    } else {
        0.0
    };

    Ok(net_premium)
}

// Complete projection
fn __complete_projection(lf: LazyFrame, sum_insured: f64) -> PolarsResult<LazyFrame> {
    // Calculate net premium
    let net_prem = ___calculate_net_premium(lf.clone(), sum_insured)?;

    // Add net premium to the lazyframe
    let lf = lf
        .with_columns(vec![
            // Claim per policy
            lit(sum_insured).alias("claim_pp"),
            // Loaded premium and round to 2 decimal places
            ((lit(1.0) + col("load_rate")) * lit(net_prem)).alias("prem_pp"),
            // Portfolio expense
            (col("exp_pp") * col("pols_if")).alias("expenses"),
        ])
        .with_columns(vec![
            // Portfolio claims
            (col("claim_pp") * col("pols_death")).alias("claims"),
            // Portfolio premiums
            (col("prem_pp") * col("pols_if")).alias("premiums"),
        ])
        .with_column(
            // Features is simple - Comission is 100% of premium in the first year
            when(col("duration").eq(0))
                .then(col("premiums"))
                .otherwise(lit(0.0))
                .alias("commissions"),
        )
        .with_column(
            (col("premiums") - col("expenses") - col("claims") - col("commissions"))
                .alias("net_cf"),
        );

    Ok(lf.collect()?.lazy()) // To avoid nested lazyframe
}

fn _project_single_model_point(
    mp: &ModelPoint,
    assumptions: &AssumptionSet,
) -> PolarsResult<DataFrame> {
    // Initialize projection dataframe - using all interger values
    let lf = __initialize_lf(mp.id, mp.term, mp.entry_age, mp.sum_insured)?;
    // Map assumptions
    let lf = __map_mort_assumption(lf, &assumptions.mort, &mp.gender)?; // Mortality assumption based on gender
    let lf = __map_other_assumption(lf, &assumptions.lapse)?; // Lapse assumption
    let lf = __map_other_assumption(lf, &assumptions.acq)?; // Acquisition expenses
    let lf = __map_other_assumption(lf, &assumptions.mtn)?; // Maintenance expenses
    let lf = __map_other_assumption(lf, &assumptions.inf)?; // Inflation assumption
    let lf = __map_other_assumption(lf, &assumptions.spot)?; // Spot rate assumption
    let lf = __map_other_assumption(lf, &assumptions.load)?; // Load rate assumption
    // Perform projection
    let lf = __discount_factor(lf)?;
    let lf = __exp_pp(lf)?;
    let lf = __policies_count(lf, mp.policy_count, mp.term)?;
    let lf = __complete_projection(lf, mp.sum_insured)?;

    Ok(lf.collect()?)
}

//---------------------------------------------------------------------------------------------------------
// PUBLIC
//---------------------------------------------------------------------------------------------------------

/*
Using the below command to run the code in parallel with limited threads finish run in 90s vs 400s in non parallel mode
The test is not exhaustive, but it shows that parallel processing can significantly speed up the projection of model points.
$env:RAYON_NUM_THREADS = 8; $env:RUST_MIN_STACK = 33554432; cargo run
*/

//----------------------------------------
// Non-parallel version of the projection
//----------------------------------------
pub fn project_single_run(run_setup: &RunSetup) -> PolarsResult<RunResult> {
    // Convert model points DataFrame to vector
    let model_points_vec = convert_model_points_df_to_vector(&run_setup.model_points_df)?;

    let mut lfs = Vec::with_capacity(model_points_vec.len());

    for mp in &model_points_vec {
        let df = _project_single_model_point(mp, &run_setup.assumptions)?;
        lfs.push(df.lazy());
    }

    // Concatenate all LazyFrames and collect to DataFrame
    let lf = concat(lfs, Default::default())?;
    let final_df = lf.collect()?;

    // Return the result with run setup and projected DataFrame
    let result = RunResult {
        run_setup: run_setup.clone(),
        projected_df: final_df,
    };

    Ok(result)
}

pub fn project_multiple_run(run_setups: &Vec<RunSetup>) -> PolarsResult<MultipleRunResult> {
    let mut results: Vec<RunResult> = Vec::with_capacity(run_setups.len()); // To collect run results

    for i in 0..run_setups.len() {
        let setup = run_setups.get(i).unwrap();
        let result = project_single_run(setup)?;
        results.push(result);
    }

    Ok(MultipleRunResult {
        run_setups: run_setups.clone(),
        run_results: results,
    })
}

//----------------------------------------
// Parallel version of the projection
//----------------------------------------
pub fn project_single_run_parallel(run_setup: &RunSetup) -> PolarsResult<RunResult> {
    // Convert model points DataFrame to vector
    let model_points_vec = convert_model_points_df_to_vector(&run_setup.model_points_df)?;

    // Process chunks of model points in parallel with limited threads
    let chunks: Vec<&[ModelPoint]> = model_points_vec.chunks(CHUNK_SIZE).collect();

    let chunk_dfs: PolarsResult<Vec<DataFrame>> = chunks
        .into_par_iter()
        .map(|chunk| {
            // Process each chunk sequentially (no nested parallelism)
            let mut lfs = Vec::new();
            for mp in chunk {
                let df = _project_single_model_point(mp, &run_setup.assumptions)?;
                lfs.push(df.lazy());
            }

            // Concatenate LazyFrames within the chunk and collect to DataFrame
            let lf = concat(lfs, Default::default())?;
            lf.collect()
        })
        .collect();

    // Unwrap the result or return the error
    let chunk_dfs = chunk_dfs?;

    // Concatenate all chunk DataFrames
    let final_df = concat(
        chunk_dfs
            .into_iter()
            .map(|df| df.lazy())
            .collect::<Vec<_>>(),
        Default::default(),
    )?
    .collect()?;

    // Return the result with run setup and projected DataFrame
    let result = RunResult {
        run_setup: run_setup.clone(),
        projected_df: final_df,
    };

    Ok(result)
}

pub fn project_multiple_run_parallel(
    run_setups: &Vec<RunSetup>,
) -> PolarsResult<MultipleRunResult> {
    let mut results: Vec<RunResult> = Vec::with_capacity(run_setups.len()); // To collect run results

    for i in 0..run_setups.len() {
        let setup = run_setups.get(i).unwrap();
        let result = project_single_run_parallel(setup)?;
        results.push(result);
    }

    Ok(MultipleRunResult {
        run_setups: run_setups.clone(),
        run_results: results,
    })
}

pub fn project_multiple_run_parallel_x2(
    run_setups: &Vec<RunSetup>,
) -> PolarsResult<MultipleRunResult> {
    let results: Vec<RunResult> = run_setups
        .par_iter()
        .map(|setup| project_single_run_parallel(setup))
        .collect::<PolarsResult<Vec<_>>>()?;

    Ok(MultipleRunResult {
        run_setups: run_setups.clone(),
        run_results: results,
    })
}
