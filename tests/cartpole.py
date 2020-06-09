from policies import FourierPolicy
from agents import FCHC
import gym
import numpy as np
import csv

# Create the environment
env = gym.make('CartPole-v0')

# Specify how to normalize the states.
def cp_normalize(obs):
    """ Normalize the features of a cartpole state state observation to the interval [0, 1]

        The feature domains are:
            [-4.8, 4.8]
            [-inf, inf]
            [-24, 24]
            [-inf, inf]

        Infinite-domain features are normalized by a sigmoid.
    """
    return np.array([
        (obs[0] + 4.8) / 9.6,
        np.exp(obs[1]) / (np.exp(obs[1]) + 1),
        (obs[2] + 24) / 48,
        np.exp(obs[3]) / (np.exp(obs[3]) + 1)
    ])

def find_okay_policy(deg, min_perf_target=10.0):
    # Start with a randomly initialized Fourier policy
    pi = FourierPolicy(deg, env.action_space.n, env.reset().size)

    # Use a simple hill-climbing agent to obtain a semi-competent policy
    agent = FCHC(pi, env, cp_normalize, gamma=1.0, sigma=18.5, n_eval_eps=50, eval_ep_len=1000)
    (mean_return, trained_policy) = agent.train_until(target_return=min_perf_target, log=True)
    print("==============================================")
    print("Found an acceptable policy with mean return {}".format(mean_return))
    print("==============================================")
    return (mean_return, trained_policy)

def generate_data(pi, n_episodes=10, max_ep_length=1000, output_path='cartpole.csv'):
    # Open a csv file to record episode data for consumption by HCPI
    with open('datasets/' + output_path, 'w', newline='') as datafile:
        w = csv.writer(datafile, delimiter=',')

        print("Writing prelude data to {}".format(output_path))
        w.writerow([pi.state_dim])
        w.writerow([pi.n_actions])
        w.writerow([pi.deg])
        w.writerow(pi.params.flatten())
        w.writerow([n_episodes])

        def run_one_ep():
            ep_return = 0.0
            obs = cp_normalize(env.reset())
            hist = list(obs)

            for t in range(max_ep_length):
                action = pi.sample(obs)
                raw_obs, r, is_terminal, _ = env.step(action)
                obs = cp_normalize(raw_obs)
                ep_return += r # Note: we're not discounting in these examples
                hist.extend([action, r]) # Record the transition data

                if is_terminal:
                    break

                hist.extend(obs) # Record the newly entered state if it's nonterminal

            w.writerow(hist)
            return hist

        # Run the newly found policy for n episodes to generate data for HCPI
        e1 = run_one_ep() # Save the history data from the first episode;

        for ep in range(1, n_episodes):
            if ep % (n_episodes // 10) == 0:
                print("Running trial episode {} of {}".format(ep + 1, n_episodes))
            run_one_ep()

        # Record the policy probabilities at each time step from the first episode
        # so that HCPI can confirm that its policy representation matches ours.
        state_dim = pi.state_dim
        policy_probs = []
        while e1:
            # Slice off one (s, a, r) triple
            step, e1 = (e1[:state_dim + 2], e1[state_dim + 2:])
            s, a = (np.array(step[:state_dim]), step[state_dim])
            policy_probs.append(pi.eval_one(s, a))

        w.writerow(policy_probs)
        env.close()


SIZES = [100000, 200000]
DEGS = [1, 2]

for n_eps in SIZES:
    for deg in DEGS:
        (mean_ret, pi) = find_okay_policy(deg, min_perf_target=100.0)
        fname = 'cartpole_deg{}_ret{}_eps{}.csv'.format(deg, round(mean_ret, 2), n_eps)
        generate_data(pi, n_episodes=n_eps, output_path=fname)

