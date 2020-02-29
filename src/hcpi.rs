extern crate rayon;
extern crate rgsl;
extern crate simplers_optimization;
use crate::data::PolicyData;
use crate::mdp::{util, FourierPolicy, History, Policy};
use rayon::prelude::*;
use rgsl::randist::t_distribution::tdist_Pinv;
use simplers_optimization::Optimizer;

/// Either a policy that has passed the safety test, or No Solution Found.
/// The f64 component of the tuple is the associated policy's score on the safety test.
/// For debugging purposes, the NSF variant wraps the failed policy parameters.
pub enum SafetyResult {
    Good((Policy, f64)),
    NSF((Policy, f64)),
}
use SafetyResult::{Good, NSF};

/// Computes the per-decision importance sampling estimator of the expected return over a
/// single history
/// NOTE: Here we are eliding the gamma^t term, because we are guaranteed gamma=1 in
/// the sample data on the specific MDP this code was written for. For this code to work on
/// general MDPs, this function would need to be modified to discount the importance-weighted
/// reward at time t by gamma^t.
fn pdis_single(hist: &History, pi_e: &FourierPolicy, pi_b: &FourierPolicy) -> f64 {
    let horizon = hist.states.len();
    let mut result = 0.0;
    for t in 0..horizon {
        let mut importance = 1.0;
        for j in 0..t {
            importance *= pi_e.eval(&hist.states[j], hist.actions[j])
                / pi_b.eval(&hist.states[j], hist.actions[j]);
        }
        result += importance * hist.rewards[t];
    }
    result
}

/// Computes the average per-decision importance sampling estimate of the expected return over
/// a dataset of many histories
fn pdis(data: &Vec<History>, pi_e: &FourierPolicy, pi_b: &FourierPolicy) -> (f64, Vec<f64>) {
    let per_hist_pdis: Vec<f64> = data
        .par_iter()
        .map(|h| pdis_single(h, pi_e, pi_b))
        .collect();
    (
        per_hist_pdis.par_iter().sum::<f64>() / data.len() as f64,
        per_hist_pdis,
    )
}

/// Computes a high-confidence lower bound on expected return of the candidate policy.
/// `delta` is the acceptable probability that the algorithm will return a policy that is
/// actually worse than the behavior policy.
fn ttest_bound(n_data: usize, pdis_mean: f64, per_hist_pdis: Vec<f64>, delta: f64) -> f64 {
    let sigma = util::sample_std(&per_hist_pdis, pdis_mean);
    let n = n_data as f64;
    (sigma / n.sqrt()) * tdist_Pinv(1.0 - delta, n)
}

/// Implements the HCPI safety test (t-test version).
/// This returns `Some(pi)` if the policy `pi` passes the safety test, otherwise `None`.
/// NOTE: `data` could be either the candidate or the safety data. It is the candidate
/// data during candidate selection, and the safety data during the final test.
pub fn safety_test(
    data: &Vec<History>,
    n_safety_data: usize,
    pi_e: &FourierPolicy,
    pi_b: &FourierPolicy,
    target: f64,
    delta: f64,
) -> SafetyResult {
    let (score, per_hist_pdis) = pdis(data, pi_e, pi_b);
    let bound = 2.0 * ttest_bound(n_safety_data, score, per_hist_pdis, delta);
    if score - bound >= target {
        Good((pi_e.theta.clone(), score))
    } else {
        NSF((pi_e.theta.clone(), score))
    }
}

/// Runs HCPI candidate selection using the Simple(x) black-box optimizer
/// Mutably borrows pi_e so that we can update the weights without having to
/// construct a new policy at every iteration.
/// The `policy_param_intervals` argument is a Vec containing the search interval for each
/// parameter of the candidate policy.
pub fn select_candidate(
    candidate_data: &PolicyData,
    pi_b: &FourierPolicy,
    n_safety_data: usize,
    target: f64,
    delta: f64,
    policy_param_intervals: Vec<(f64, f64)>,
) -> (f64, Policy) {
    // Define the objective function we're trying to maximize.
    // We want a solution in argmax over theta PDIS(D_c, theta, pi_b), subject to
    // the constraint that we expect theta to pass the safety test.
    // So, we use a barrier function that penalizes NSF values of theta.
    // The barrier function has the PDIS estimate added as a shaping term to encourage
    // movement toward higher PDIS values.
    let candidate_obj_fn = |theta_e: &[f64]| {
        let pi_e = FourierPolicy::new(
            candidate_data.fourier_deg,
            candidate_data.state_dim,
            theta_e.to_vec(),
            candidate_data.n_actions,
        );
        match safety_test(
            &candidate_data.hists,
            n_safety_data,
            &pi_e,
            pi_b,
            target,
            delta,
        ) {
            Good((_, score)) => score,
            // Use the PDIS estimate as a shaping term when no solution is found
            NSF((_, score)) => -100000.0 + score,
        }
    };

    let n_iters = 200;
    let (optimum, candidate_params) =
        Optimizer::maximize(&candidate_obj_fn, &policy_param_intervals, n_iters);
    (optimum, candidate_params.into_vec())
}
