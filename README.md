# Overview

This is a Rust crate implementing a variant of [High-Confidence Policy Improvement](http://proceedings.mlr.press/v37/thomas15.pdf).
HCPI is a reinforcement learning algorithm that takes trajectories generated by a behavior policy and uses them to recommend a new policy that is better with high probability.
The acceptable probability of a regression in policy performance is an input to the algorithm that can be tuned by the user.
The intent of such algorithms is to allow safe policy improvement in domains (e.g. medicine) where a competent behavior policy is known, but on-policy exploration is prohibited because mistakes are extremely costly.

This implementation works by using black-box optimization to find policy parameters that optimize expected discounted return, subject to the constraint that they are expected to outperform the behavior policy with high probability.
Expected returns for candidate policies are estimated using per-decision importance sampling to reweight the observations collected under the behavior policy.
To ensure it really is likely to be an improvement, the resulting candidate policy is then subjected to a safety test using a holdout split from the input data, and is either returned or discarded depending on the outcome.

The `src/{mdp, data, hcpi}.rs` files define an interface for running HCPI, and `src/main.rs` contains an example of using HCPI to generate 100 improved policies for a small MDP using the behavior policy data in `data.csv`.

This code was written in Dec., 2019, as a project for CS 687 (Reinforcement Learning) at UMass Amherst, and released with permission from the instructor.

See also:
- [Phil Thomas's 2019 RL Course Notes](https://people.cs.umass.edu/~pthomas/courses/CMPSCI_687_Fall2019/687_F19.pdf)
- [UMass AI Safety Page](https://aisafety.cs.umass.edu/)

# Input Data Format
The input data is stored in a top-level file called `data.csv`, whose rows contain:
1. The number of state features.
2. The number of actions.
3. The Fourier basis order used by the behavior policy.
4. The parameters of the behavior policy.
5. The number of episodes N of data generated under the behavior policy.
6. N rows of numbers, where each row indicates the full history of an episode.
7. A list of (state, action) probabilities generated by the behavior policy, used to test that the HCPI policy representation is accurate.

See p. 151 of the [course notes](https://people.cs.umass.edu/~pthomas/courses/CMPSCI_687_Fall2019/687_F19.pdf) for full details.

# Steps to Run
1. [Install Rust.](https://www.rust-lang.org/tools/install)
2. This project depends on FFI bindings into the GNU Scientific Library. Most package managers bundle GSL, so installing it should be painless. A scary compile error from the HCPI code probably means that rustc can't find the GSL.
3. (Optional) Run `cargo test` to test that the data file is internally consistent and the HCPI policy representation matches the behavior policy representation.
4. In the top level of the source directory (the level containing `Cargo.toml`), run `cargo run --release`. The crate should compile without warnings.


Note: In the working directory, `main.rs` will create a top-level directory called `output`, which will be populated with improved policies as they are found. The code will panic and prompt you to delete the `output` directory if it already exists, in order to avoid overwriting improved policies from a previous run.

# Limitations
- While HCPI works in a more general setting, this code only handles Fourier policies over finite action spaces.
- This code was written to solve a specific problem, and makes no attempt to provide a general library API.
- Consequently, some hyperparameters or constants may be hard-coded, though this should be mostly confined to `main.rs`.

