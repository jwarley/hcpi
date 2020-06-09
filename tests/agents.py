from policies import FourierPolicy
import numpy as np

class FCHC:
    def __init__(self, init_policy, env, norm_fn, gamma=0.9, sigma=0.1, n_eval_eps=20, eval_ep_len=1000):
        self.pi = init_policy
        self.sigma = sigma
        self.n_eval_eps = n_eval_eps
        self.eval_ep_len = eval_ep_len
        self.env = env
        self.gamma = gamma
        self.normalize = norm_fn

    def evaluate(self, pi, N, max_ep_len, gamma, log):
        """ Compute the Monte Carlo return over N episodes under policy `pi` """
        if log:
            print("Evaluating new policy...")

        ep_returns = np.zeros(N)

        for ep in range(N):
            # Run one episode under the evaluation policy
            ep_return = 0.0
            obs = self.normalize(self.env.reset())

            for t in range(max_ep_len):
                action = pi.sample(obs)
                raw_obs, r, is_terminal, _ = self.env.step(action)
                ep_return += gamma**t * r
                obs = self.normalize(raw_obs)

                if is_terminal:
                    break
            ep_returns[ep] = ep_return

        mean_return = np.mean(ep_returns)
        if log:
            print("Mean return for evaluation policy: {}".format(mean_return))
        return mean_return

    def train(self, log=False):
        best_avg_return = self.evaluate(self.pi, self.n_eval_eps, self.eval_ep_len, log, self.gamma)

        while True:
            # Sample a new candidate policy from a normal distribution
            mu = self.pi.get_params().flatten()
            sigma = self.sigma * np.eye(mu.size)
            candidate_params = np.random.default_rng().multivariate_normal(mu, sigma) \
                .reshape(self.pi.get_params().shape)
            candidate_pi = FourierPolicy(
                self.pi.deg,
                self.pi.n_actions,
                self.pi.state_dim
            ).with_params(candidate_params)

            # Evaluate the candidate policy and update the behavior policy if it is outperformed
            candidate_return = self.evaluate(candidate_pi, self.n_eval_eps, self.eval_ep_len, log, self.gamma)
            if candidate_return > best_avg_return:
                self.pi = candidate_pi
                best_avg_return = candidate_return
                if log:
                    print("new best return: {}".format(best_avg_return))
                    print("params: {}".format(self.pi.get_params()))

    def train_until(self, target_return=np.Inf, log=False):
        best_avg_return = self.evaluate(self.pi, self.n_eval_eps, self.eval_ep_len, log, self.gamma)

        while best_avg_return < target_return:
            # Sample a new candidate policy from a normal distribution
            mu = self.pi.get_params().flatten()
            sigma = self.sigma * np.eye(mu.size)
            candidate_params = np.random.default_rng().multivariate_normal(mu, sigma) \
                .reshape(self.pi.get_params().shape)
            candidate_pi = FourierPolicy(
                self.pi.deg,
                self.pi.n_actions,
                self.pi.state_dim
            ).with_params(candidate_params)

            # Evaluate the candidate policy and update the behavior policy if it is outperformed
            candidate_return = self.evaluate(candidate_pi, self.n_eval_eps, self.eval_ep_len, log, self.gamma)
            if candidate_return > best_avg_return:
                self.pi = candidate_pi
                best_avg_return = candidate_return
                if log:
                    print("new best return: {}".format(best_avg_return))
                    print("params: {}".format(self.pi.get_params()))

        return (best_avg_return, self.pi)



