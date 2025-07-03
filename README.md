# BASIC TERM SM MODEL IN RUST FROM LIFELIB

## About The Project

Some notes:

- The main reference is [Basic term S](https://lifelib.io/libraries/basiclife/BasicTerm_S.html). Model S, M and SC are basically the same.
- term_sm.ipynb Jupyter Notebook is only for reference, not an actual implementation. There is a limitation with Rust Kernel in Jupyter lab that lazyframe technology cannot work properly for querying.

What can be taken out from this repo?

- The project introduces the implementation of Rust language and its latest dataframe technology to support actuarial work.
- The project also shows that spreadsheet, although being convinient, is not the best suited tool to handle full projections which might consist of millions records even for the most basic products.


## Getting started

### Rust installation

Install [Rust](https://www.rust-lang.org/)

### Repo installation

- Clone the repo:

```shell
git clone https://github.com/hnlearndev/Basic-Term-Pricing-Model-Rust-lifelibBasicTermSM
```

### Usage

- Go to the local repo and build dependencies:

```shell
cargo build
```

- Run the package

```shell
cargo run
```

## Roadmap

- Add Changelog
- Add premium rate plot and introduce technique (eg: spline...)  to smooth curve.

## Contact

Trung-Hieu Nguyen - [hieunt.hello@gmail.com](mailto:hieunt.hello@gmail.com)

Project Link: [https://github.com/hnlearndev/Basic-Term-Pricing-Model-Rust-lifelibBasicTermSM](https://github.com/hnlearndev/Basic-Term-Pricing-Model-Rust-lifelibBasicTermSM.git)
