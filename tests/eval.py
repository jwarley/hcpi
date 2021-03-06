from policies import FourierPolicy
from agents import FCHC
import gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import csv
import os

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

# Returns a list of `n_eps` episode returns under the policy `theta`
def test_params(env, theta, fourier_deg, n_eps=100, max_ep_length=1000):
    state_dim = env.reset().size
    pi = FourierPolicy(fourier_deg, env.action_space.n, state_dim) \
        .with_params(theta.reshape((n_actions, (fourier_deg + 1)**state_dim)))

    def run_one_ep(env):
        ep_return = 0.0
        obs = cp_normalize(env.reset())

        for t in range(max_ep_length):
            action = pi.sample(obs)
            raw_obs, r, is_terminal, _ = env.step(action)
            obs = cp_normalize(raw_obs)
            ep_return += r # Note: we're not discounting in these examples

            if is_terminal:
                break

        return ep_return

    rets = []
    for ep in range(n_eps):
        rets.append(run_one_ep(env))

    return rets


# Create the environment
env = gym.make('CartPole-v0')

# Read the behavior policy from the data.csv file

# A list that will contain the behavior policy parameters,
# followed by the parameters for every policy we want to test
params_to_test = []

print("Reading behavior policy")
(state_dim, n_actions, fourier_deg) = (None, None, None)
with open('data.csv') as datafile:
    r = csv.reader(datafile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
    state_dim = int(next(r)[0])
    n_actions = int(next(r)[0])
    fourier_deg = int(next(r)[0])
    params_to_test.append(np.array(next(r)))

print("Checking consistency with environment...")
assert state_dim == env.reset().size
assert n_actions == env.action_space.n

print("Reading test policies...")
for filename in os.listdir("./output"):
    with open("./output/" + filename) as datafile:
        r = csv.reader(datafile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
        params_to_test.append(np.array(next(r)))

print("Conducting tests...")
results = []
with open('tests/eval.csv', 'w') as datafile:
    w = csv.writer(datafile, delimiter=',')
    n = 0
    for theta in params_to_test:
        if n == 0:
            print("Testing behavior policy.")
        else:
            print("Testing proposal {}.".format(n))
        rets = test_params(env, theta, fourier_deg)
        results.append(rets)
        w.writerow(rets)
        n += 1


fig, ax = plt.subplots()
t = range(1, 101)

# Plot the behavior policy data
ax.hlines(y=np.mean(results[0]), xmin=1, xmax=100, label="Behavior Mean", colors='tab:blue', linestyles='dashed')
ax.plot(t, results[0], 'o', color='tab:blue', label="Behavior")

# Plot the proposal policies
ax.plot(t, results[1], '^', label="Proposal", color="tab:red")
for i in range(2, len(results)):
    ax.plot(t, results[i], '^', color="tab:red")

ax.set(
    xlabel="Episode",
    ylabel="Return",
    title="Behavior vs. HCPI Proposal Policies on CartPole"
)
ax.grid()
ax.legend(loc="lower left")

fig.savefig("eval.png")
plt.show()
