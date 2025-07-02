use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand::rngs::StdRng;
use ndarray_rand::rand_distr::Uniform;
use polars::prelude::*;

pub fn test_term() {
    // Number of model points
    let mp_size = 10000;

    // Use a fixed seed for reproducibility
    let mut rng = StdRng::seed_from_u64(9876);

    // Issue Age (Integer): Random 20 - 59 year old
    let entry_age = Array1::random_using(mp_size, Uniform::new(20, 60), &mut rng); // 60 is exclusive, so range is 20-59

    // Gender (String): Random "M" and "F"
    let gender_binary = Array1::random_using(mp_size, Uniform::new(0, 2), &mut rng); // 0 or 1
    let gender: Vec<&str> = gender_binary
        .iter()
        .map(|&x| if x == 0 { "M" } else { "F" }) // map 0 to "M" and 1 to "F"
        .collect();

    // Policy term (Integer): Random 10, 15 or 20
    let term = (Array1::random_using(mp_size, Uniform::new(3, 6), &mut rng)) * 5;

    // Policy count
    let policy_count = Array1::<i32>::ones(mp_size);

    // Sum insured (Float): Random values between 100,000 and 1,000,000 (multiple of 1000)
    let sum_insured =
        Array1::random_using(mp_size, Uniform::new(0.0f64, 1.0f64), &mut rng) // Random floats between 0 and 1
            .mapv(|x| (((900_000.0 * x + 100_000.0) / 1000.0).round() * 1000.0) as f64);

    // Create a DataFrame with the generated data
    let df = df![
        "entry_age" => entry_age.to_vec(),
        "gender" => gender,
        "term" => term.to_vec(),
        "policy_count" => policy_count.to_vec(),
        "sum_insured" => sum_insured.to_vec(),
    ]
    .unwrap();

    // Display the DataFrame
    println!("{:?}", df);
}
