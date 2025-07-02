use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand::rngs::StdRng;
use ndarray_rand::rand_distr::Uniform;
use polars::prelude::*;

// These functions should be placed in the same module (eg: assumptions.rs)
// NOTE: DO NOT MODIFY THIS

pub struct ModelPoint {
    pub id: i32,
    pub entry_age: i32,
    pub gender: String,
    pub term: i32,
    pub policy_count: f64,
    pub sum_insured: f64,
}

pub fn generate_model_points(mp_size: usize, seed: usize) -> PolarsResult<DataFrame> {
    // Get seed for random number generation
    let mut rng = StdRng::seed_from_u64(seed as u64);

    // Issue Age (Integer): Random 20 - 59 year old
    let entry_age = Array1::random_using(mp_size, Uniform::new(20, 60), &mut rng); // 60 is exclusive, so range is 20-59

    // Gender (String): Random "M" and "F"
    let gender_binary = Array1::random_using(mp_size, Uniform::new(0, 2), &mut rng); // 0 or 1
    let gender: Vec<&str> = gender_binary
        .iter()
        .map(|&x| if x == 0 { "M" } else { "F" }) // map 0 to "M" and 1 to "F"
        .collect();

    // Policy term (Integer): Random 10, 15 or 20
    let term = (Array1::random_using(mp_size, Uniform::new(2, 5), &mut rng)) * 5;

    // Policy count
    let policy_count = Array1::<f64>::ones(mp_size);

    // Sum insured (Float): Random values between 100,000 and 1,000,000 (multiple of 1000)
    let sum_insured =
        Array1::random_using(mp_size, Uniform::new(0.0f64, 1.0f64), &mut rng) // Random floats between 0 and 1
            .mapv(|x| (((900_000.0 * x + 100_000.0) / 1000.0).round() * 1000.0) as f64);

    // Create a DataFrame with the generated data
    let model_points_df = df![
        "id"  => (1..(mp_size+1) as i32).collect::<Vec<i32>>(),
        "entry_age" => entry_age.to_vec(),
        "gender" => gender,
        "term" => term.to_vec(),
        "policy_count" => policy_count.to_vec().into_iter().map(|x| x as f64).collect::<Vec<f64>>(),
        "sum_insured" => sum_insured.to_vec(),
    ]?;

    Ok(model_points_df)
}

pub fn convert_model_points_df_to_vector(df: &DataFrame) -> PolarsResult<Vec<ModelPoint>> {
    let id = df.column("id")?.i32()?;
    let entry_age = df.column("entry_age")?.i32()?;
    let gender = df.column("gender")?.str()?;
    let term = df.column("term")?.i32()?;
    let policy_count = df.column("policy_count")?.f64()?;
    let sum_insured = df.column("sum_insured")?.f64()?;

    let model_points = (0..df.height())
        .map(|i| ModelPoint {
            id: id.get(i).unwrap(),
            entry_age: entry_age.get(i).unwrap(),
            gender: gender.get(i).unwrap().to_string(),
            term: term.get(i).unwrap(),
            policy_count: policy_count.get(i).unwrap(),
            sum_insured: sum_insured.get(i).unwrap(),
        })
        .collect();

    Ok(model_points)
}
