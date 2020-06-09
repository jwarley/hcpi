mod data;
mod hcpi;
mod mdp;
use data::PolicyData;
use hcpi::{safety_test, select_candidate, SafetyResult};
use mdp::FourierPolicy;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    println!("Creating output directories...");
    std::fs::create_dir("./output").expect(
        "Error: `output` directory already exists. Please remove it before generating new polcies.",
    );
    std::fs::create_dir("./failed").expect(
        "Error: `failed` directory already exists. Please remove it before generating new polcies.",
    );

    // Read in the behavior policy data from a file and construct our policy representation.
    let data_path = "data.csv";
    println!("Loading behavior policy data from {}", data_path);
    let d = PolicyData::from_file(data_path);

    println!("Constructing behavior policy...");
    let pi_b = FourierPolicy::new(d.fourier_deg, d.state_dim, d.pi_b.clone(), d.n_actions);

    // Generate `N_DESIRED_POLICIES` improved policies.
    // This code will not count candidate policies that fail the safety test, i.e. it will run
    // until N actual improved policies are found. Consequently, if the behavior policy is close
    // to optimal or the amount of behavior data is small, this loop may run indefinitely.
    // The number of improved policies found will be printed at each search iteration so that the
    // search progress can be monitored and halted if it's clear that improvements on the behavior
    // policy are rare.
    const N_DESIRED_POLICIES: usize = 10;
    let mut search_iter = 0;
    let mut policies_found = 0;
    let mut failed = 0;

    while policies_found < N_DESIRED_POLICIES {
        println!("===========================");
        println!("Beginning new policy search");
        println!("===========================\n");
        println!("Search iteration: {}", search_iter);
        println!("Passing policies found: {}\n", policies_found);

        println!("Randomly splitting data into safety and candidate sets...");
        let (d_c, d_s) = d.random_split(d.num_eps / 2);
        let num_cand = d_c.num_eps;
        let num_safety = d_s.num_eps;
        println!("Size of candidate data: {}", num_cand);
        println!("Size of safety data: {}\n", num_safety);

        // We want a 90% chance of getting a better policy than pi_b
        let (target, delta) = (d.pi_b_avg_return(), 0.10);

        // I'm making an assumption that we can do pretty well by searching over a bounded interval
        // of parameter space containing the original policy parameters.
        // I'm not aware of a principled way to choose the search interval for each policy
        // parameter in a general problem setting.
        let search_intervals = vec![(-50., 50.); pi_b.theta.len()];
        let n_optim_iters = 500; // Number of iterations to run the black-box optimizer

        println!("Searching for improved policy parameters...");
        let (opt_val, theta_opt) = select_candidate(
            &d_c,
            &pi_b,
            num_safety,
            target,
            delta,
            search_intervals,
            n_optim_iters,
        );
        println!("Found a candidate with score {}", opt_val);

        println!("Running safety test on found policy...");
        let pi_opt = FourierPolicy::new(d.fourier_deg, d.state_dim, theta_opt, d.n_actions);
        match safety_test(&d_s.hists, num_safety, &pi_opt, &pi_b, target, delta, false) {
            SafetyResult::Good((theta, score)) => {
                println!("New policy passed safety test! (score {})", score);
                // Write the policy to a csv file
                let mut wtr = csv::Writer::from_path(format!("output/{}.csv", policies_found + 1))?;
                wtr.serialize(theta.to_vec())?;
                wtr.flush()?;
                policies_found += 1;
            }
            SafetyResult::NSF((theta, score)) => {
                println!(
                    "The found policy failed the safety test on the safety data. (score {})",
                    score
                );
                let mut wtr = csv::Writer::from_path(format!("failed/{}.csv", failed + 1))?;
                wtr.serialize(theta.to_vec())?;
                wtr.flush()?;
                failed += 1;
            }
        }

        search_iter += 1;
    }
    Ok(())
}
