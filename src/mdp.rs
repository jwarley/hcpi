extern crate convert_base;
extern crate ndarray;
use convert_base::Convert;
use ndarray::prelude::*;
use std::f64::consts::{E, PI};

pub mod util {
    extern crate rayon;
    use rayon::prelude::*;

    /// Unbiased sample standard deviation of a vector xs with mean x_bar
    pub fn sample_std(xs: &Vec<f64>, x_bar: f64) -> f64 {
        let sum_squares: f64 = xs.par_iter().map(|x| (x - x_bar).powf(2.0)).sum();
        (sum_squares / (xs.len() as f64 - 1.0)).sqrt()
    }
}

pub type MDPState = Array1<f64>;
pub type Policy = Array1<f64>;
pub type Action = u32;

#[derive(Debug, Clone)]
pub struct History {
    pub states: Vec<MDPState>,
    pub actions: Vec<Action>,
    pub rewards: Vec<f64>,
}

#[derive(Debug)]
pub struct FourierPolicy {
    pub k: u32,         // The order of the Fourier basis
    pub m: u32,         // The number of state features
    pub theta: Policy,  // The policy parameters
    pub n_actions: u32, // Actions are 0, 1, ..., n_actions
    // The Fourier basis coefficients. These will be
    // integers but are f64 to avoid casting in dot product
    coeffs: Array2<f64>,
}

impl FourierPolicy {
    pub fn new(k: u32, m: u32, theta: Policy, n_actions: u32) -> Self {
        // coeffs should have size (k+1)^m, where coeffs[i] is a Vec<u8>
        // representing the little-endian base-(k+1) encoding of i
        let mut base_conv = Convert::new(10, (k + 1) as u64);

        let mut c: Vec<Vec<u32>> = (0..((k + 1).pow(m as u32)))
            .map(|x: u32| vec![x]) // converter takes vecs
            .map(|d: Vec<u32>| base_conv.convert::<u32, u32>(&d))
            .collect();

        // Extend all the c_i to be full-rank
        for c_i in &mut c {
            c_i.resize(m as usize, 0);
        }

        let mut coeffs: Array2<f64> = Array2::zeros(((k + 1).pow(m) as usize, m as usize));
        for (i, c_i) in c.iter().enumerate() {
            for (j, el) in c_i.iter().enumerate() {
                coeffs[[i, j]] = *el as f64;
            }
        }

        FourierPolicy {
            k,
            m,
            theta,
            n_actions,
            coeffs,
        }
    }

    /// Returns the Fourier features for the state `s`
    pub fn basify(&self, s: &MDPState) -> Array1<f64> {
        self.coeffs.dot(s).mapv(|x| f64::cos(PI * x))
    }

    /// Returns `theta_a`, the parameter vector for action `a`
    fn get_action_params(&self, a: Action) -> ArrayView1<f64> {
        // There will be |A|(k+1)^m numbers in self.theta
        // The `a`th group of (k+1)^m numbers is theta_a
        let block_len: u32 = (self.k + 1).pow(self.m as u32) as u32;
        let start = a * block_len;
        let end = (start + block_len) as usize;
        self.theta.slice(s![start as usize..end])
    }

    /// Returns pi(s, a) using softmax action selection
    pub fn eval(&self, s: &MDPState, a: Action) -> f64 {
        let phi_s = &self.basify(s);
        let numerator = E.powf(phi_s.dot(&self.get_action_params(a)));
        let mut denom = 0.0;
        for actn in 0..(self.n_actions) {
            denom += E.powf(phi_s.dot(&self.get_action_params(actn)));
        }
        numerator / denom
    }
}

impl History {
    fn new() -> Self {
        History {
            states: Vec::new(),
            actions: Vec::new(),
            rewards: Vec::new(),
        }
    }

    /// Parse a history from csv.
    pub fn from_seq(seq: csv::StringRecord, state_dim: u32) -> Self {
        let mut h = History::new();

        let mut episode: Vec<f64> = seq
            .into_iter()
            .map(|str| str.trim().parse().unwrap())
            .collect();

        while episode.len() != 0 {
            // Push the state s_t
            let (s_t, rest) = episode.split_at(state_dim as usize);
            h.states.push(Array::from(s_t.to_vec()));
            episode = rest.to_vec();

            // Push the action a_t
            if let Some((a_t, rest)) = episode.split_first() {
                h.actions.push(*a_t as u32);
                episode = rest.to_vec();
            }

            // Push the reward r_t
            if let Some((r_t, rest)) = episode.split_first() {
                h.rewards.push(*r_t);
                episode = rest.to_vec();
            }
        }

        h
    }
}
