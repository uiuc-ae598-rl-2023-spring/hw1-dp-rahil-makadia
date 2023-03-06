import numpy as np
np.set_printoptions(precision=8, linewidth=np.inf)
import matplotlib.pyplot as plt
import discrete_pendulum

def wrap_pi(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi

def wrap_2pi(x):
    return x % (2 * np.pi)

wrap_func = wrap_pi

def test_x_to_s(env):
    theta = np.linspace(-np.pi * (1 - (1 / env.n_theta)), np.pi * (1 - (1 / env.n_theta)), env.n_theta)
    thetadot = np.linspace(-env.max_thetadot * (1 - (1 / env.n_thetadot)), env.max_thetadot * (1 - (1 / env.n_thetadot)), env.n_thetadot)
    for s in range(env.num_states):
        i = s // env.n_thetadot
        j = s % env.n_thetadot
        s1 = env._x_to_s([theta[i], thetadot[j]])
        if s1 != s:
            raise ValueError(f'test_x_to_s: error in state representation: {s} and {s1} should be the same')
    return None

def epsilon_greedy_action(Q, epsilon, s, return_best_action=False):
    random_action = np.random.uniform(0,1)
    return np.argmax(Q[s]) if random_action > epsilon else np.random.randint(0, Q.shape[1])

def get_policy(Q, env):
    # initialize policy
    pi = np.zeros(env.num_states)
    # Choose A from S using policy derived from Q (e.g., "-greedy)
    for s in range(env.num_states):
        q_max_prev = None
        action_good = [] # list of actions with same q_max
        for a in range(env.num_actions):
            if q_max_prev is None or Q[s, a] > q_max_prev:
                action_good = [a]
                q_max_prev = Q[s, a]
            elif Q[s, a] == q_max_prev:
                action_good.append(a)
        good_prob = 1 / len(action_good)
        pi[s] = np.argmax([good_prob if a in action_good else 0 for a in range(env.num_actions)])
    return pi

def pendulum_sarsa(n_theta, n_thetadot, n_tau, alpha, epsilon, num_episodes=100, gamma=0.95, verbose=False, plot=False):
    # sourcery skip: extract-duplicate-method
    if verbose:
        print('------------- SARSA -------------')
        print(f'n_theta: {n_theta}, n_thetadot: {n_thetadot}, n_tau: {n_tau}, alpha: {alpha}, epsilon: {epsilon}, num_episodes: {num_episodes}, gamma: {gamma}')
    
    # Create environment
    env = discrete_pendulum.Pendulum(n_theta=n_theta, n_thetadot=n_thetadot, n_tau=n_tau)
    
    # Apply unit test to check state representation
    test_x_to_s(env)
    # Initialize simulation
    s = env.reset()

    # apply SARSA to find optimal policy and value function
    # initialize value function
    Q = np.zeros((env.num_states, env.num_actions))
    log = {'G': []}
    pi = np.zeros(env.num_states)
    for _ in range(num_episodes):
        Q_old = Q.copy()
        # Initialize simulation
        s = env.reset()
        log['t'] = [0]
        log['s'] = [s]
        log['a'] = []
        log['r'] = []
        log['theta'] = [wrap_func(env.x[0])]
        log['thetadot'] = [env.x[1]]
        G = 0
        a = epsilon_greedy_action(Q, epsilon, s, return_best_action=False)
        # Simulate until episode is done
        done = False
        n_iter = 0
        while not done:
            s1, r, done = env.step(a)
            a1 = epsilon_greedy_action(Q, epsilon, s1, return_best_action=False)
            Q[s, a] += alpha * (r + gamma * Q[s1, a1] - Q[s, a])
            s = s1
            a = a1
            log['t'].append(log['t'][-1] + 1)
            log['s'].append(s)
            log['a'].append(a)
            log['r'].append(r)
            log['theta'].append(wrap_func(env.x[0]))
            log['thetadot'].append(env.x[1])
            G += r*gamma**n_iter
            n_iter += 1
        log['G'].append(G)
    pi = get_policy(Q, env)
    # pi = np.argmax(Q, axis=1)
    if verbose:
        print(f'Q: \n {Q}')
        print(f'pi: \n {pi.reshape(n_theta, n_thetadot)}')
    if plot:
        # Plot data and save to png file
        plt.figure(figsize=(18, 5), dpi=100)
        plt.subplot(1, 3, 1)
        plt.plot(log['t'], log['s'], label='s')
        plt.plot(log['t'][:-1], log['a'], label='a')
        plt.plot(log['t'][:-1], log['r'], label='r')
        plt.xlabel('t')
        plt.ylabel('State / Action / Reward')
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(log['t'], log['theta'], label=r'$\theta$')
        plt.axhline(-np.pi, color='r', linestyle='--')
        plt.axhline(np.pi, color='r', linestyle='--', label=r'$\theta=\pm\pi$')
        plt.axhline(-0.1*np.pi, color='g', linestyle='--')
        plt.axhline(0.1*np.pi, color='g', linestyle='--', label=r'$\theta=\pm0.1\pi$')
        plt.plot(log['t'], log['thetadot'], label=r'$\dot{\theta}$')
        plt.xlabel('t')
        plt.ylabel('theta / thetadot')
        plt.legend()

        plt.subplot(1, 3, 3)
        plt.plot(np.arange(num_episodes), log['G'], '.', label='G')
        plt.xlabel('Episode #')
        plt.ylabel('Return')
        plt.yscale('symlog')
        plt.legend()
        plt.savefig('figures/pendulum/traj_return_sarsa.png', bbox_inches='tight')
    return Q, pi, log

def pendulum_q_learning(n_theta, n_thetadot, n_tau, alpha, epsilon, num_episodes=100, gamma=0.95, verbose=False, plot=False):
    # sourcery skip: extract-duplicate-method
    if verbose:
        print('------------- Q-Learning -------------')
        print(f'n_theta: {n_theta}, n_thetadot: {n_thetadot}, n_tau: {n_tau}, alpha: {alpha}, epsilon: {epsilon}, num_episodes: {num_episodes}, gamma: {gamma}')
    
    # Create environment
    env = discrete_pendulum.Pendulum(n_theta=n_theta, n_thetadot=n_thetadot, n_tau=n_tau)
    # Apply unit test to check state representation
    test_x_to_s(env)
    # Initialize simulation
    s = env.reset()

    # apply Q-Learning to find optimal policy and value function
    # initialize value function
    Q = np.zeros((env.num_states, env.num_actions))
    log = {'G': []}
    pi = np.zeros(env.num_states)
    for _ in range(num_episodes):
        Q_old = Q.copy()
        # Initialize simulation
        s = env.reset()
        log['t'] = [0]
        log['s'] = [s]
        log['a'] = []
        log['r'] = []
        log['theta'] = [wrap_func(env.x[0])]
        log['thetadot'] = [env.x[1]]
        G = 0
        # Simulate until episode is done
        done = False
        n_iter = 0
        while not done:
            a = epsilon_greedy_action(Q, epsilon, s, return_best_action=False)
            s1, r, done = env.step(a)
            Q[s, a] += alpha * (r + gamma * np.max(Q[s1]) - Q[s, a])
            s = s1
            log['t'].append(log['t'][-1] + 1)
            log['s'].append(s)
            log['a'].append(a)
            log['r'].append(r)
            log['theta'].append(wrap_func(env.x[0]))
            log['thetadot'].append(env.x[1])
            G += r*gamma**n_iter
            n_iter += 1
        log['G'].append(G)
    pi = get_policy(Q, env)
    # pi = np.argmax(Q, axis=1)
    if verbose:
        print(f'Q: \n {Q}')
        print(f'pi: \n {pi.reshape(n_theta, n_thetadot)}')
    if plot:
        # Plot data and save to png file
        plt.figure(figsize=(18, 5), dpi=100)
        plt.subplot(1, 3, 1)
        plt.plot(log['t'], log['s'], label='s')
        plt.plot(log['t'][:-1], log['a'], label='a')
        plt.plot(log['t'][:-1], log['r'], label='r')
        plt.xlabel('t')
        plt.ylabel('State / Action / Reward')
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(log['t'], log['theta'], label=r'$\theta$')
        plt.axhline(-np.pi, color='r', linestyle='--')
        plt.axhline(np.pi, color='r', linestyle='--', label=r'$\theta=\pm\pi$')
        plt.axhline(-0.1*np.pi, color='g', linestyle='--')
        plt.axhline(0.1*np.pi, color='g', linestyle='--', label=r'$\theta=\pm0.1\pi$')
        plt.plot(log['t'], log['thetadot'], label=r'$\dot{\theta}$')
        plt.xlabel('t')
        plt.ylabel('theta / thetadot')
        plt.legend()

        plt.subplot(1, 3, 3)
        plt.plot(np.arange(num_episodes), log['G'], '.', label='G')
        plt.xlabel('Episode #')
        plt.ylabel('Return')
        plt.yscale('symlog')
        plt.legend()
        plt.savefig('figures/pendulum/traj_return_q_learning.png', bbox_inches='tight')
    return Q, pi, log

def pendulum_td0(n_theta, n_thetadot, n_tau, pi, alpha, num_episodes=100, gamma=0.95, verbose=False, method='', plot=False):
    # sourcery skip: extract-duplicate-method
    if verbose:
        print('------------- TD(0) -------------')
        print(f'n_theta: {n_theta}, n_thetadot: {n_thetadot}, n_tau: {n_tau}, alpha: {alpha}, num_episodes: {num_episodes}, gamma: {gamma}')
    # Create environment
    env = discrete_pendulum.Pendulum(n_theta=n_theta, n_thetadot=n_thetadot, n_tau=n_tau)
    # Apply unit test to check state representation
    test_x_to_s(env)

    # apply TD(0) to find optimal value function
    # initialize value function
    V = np.zeros(env.num_states)
    log = {'G': []}
    for _ in range(num_episodes):
        # Initialize simulation
        s = env.reset()
        log['t'] = [0]
        log['s'] = [s]
        log['a'] = []
        log['r'] = []
        log['theta'] = [wrap_func(env.x[0])]
        log['thetadot'] = [env.x[1]]
        G = 0
        # Simulate until episode is done
        done = False
        n_iter = 0
        while not done:
            a = pi[s]
            s1, r, done = env.step(a)
            V[s] += alpha * (r + gamma * V[s1] - V[s])
            s = s1
            log['t'].append(log['t'][-1] + 1)
            log['s'].append(s)
            log['a'].append(a)
            log['r'].append(r)
            log['theta'].append(wrap_func(env.x[0]))
            log['thetadot'].append(env.x[1])
            G += r*gamma**n_iter
            n_iter += 1
        log['G'].append(G)
    if verbose:
        print(f'V: \n {V.reshape(n_theta, n_thetadot)}')
    # Plot data and save to png file
    if plot:
        # Plot data and save to png file
        plt.figure(figsize=(18, 5), dpi=100)
        plt.subplot(1, 3, 1)
        plt.plot(log['t'], log['s'], label='s')
        plt.plot(log['t'][:-1], log['a'], label='a')
        plt.plot(log['t'][:-1], log['r'], label='r')
        plt.xlabel('t')
        plt.ylabel('State / Action / Reward')
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(log['t'], log['theta'], label=r'$\theta$')
        plt.axhline(-np.pi, color='r', linestyle='--')
        plt.axhline(np.pi, color='r', linestyle='--', label=r'$\theta=\pm\pi$')
        plt.axhline(-0.1*np.pi, color='g', linestyle='--')
        plt.axhline(0.1*np.pi, color='g', linestyle='--', label=r'$\theta=\pm0.1\pi$')
        plt.plot(log['t'], log['thetadot'], label=r'$\dot{\theta}$')
        plt.xlabel('t')
        plt.ylabel('theta / thetadot')
        plt.legend()

        plt.subplot(1, 3, 3)
        plt.plot(np.arange(num_episodes), log['G'], '.', label='G')
        plt.xlabel('Episode #')
        plt.ylabel('Return')
        plt.yscale('symlog')
        plt.legend()
        plt.savefig(f'figures/pendulum/traj_return_td0_{method}.png', bbox_inches='tight')
    return V, log

def plot_diff_epsilon_alpha(n_theta, n_thetadot, n_tau, method, num_episodes=100, gamma=0.95):
    # sourcery skip: extract-duplicate-method
    alpha_nom = 0.5
    alpha_min = 0.1
    alpha_max = 1.0
    alpha_step = 0.1
    alpha_num_steps = int((alpha_max-alpha_min)/alpha_step)+1
    alpha_arr = np.linspace(alpha_min, alpha_max, alpha_num_steps)

    epsilon_nom = 0.8
    epsilon_min_power = -10
    epsilon_max_power = -1
    epsilon_num_steps = int(epsilon_max_power - epsilon_min_power) + 1
    epsilon_arr = np.logspace(epsilon_min_power, epsilon_max_power, epsilon_num_steps)

    plt.figure(figsize=(12, 5), dpi=100)
    plt.subplot(1, 2, 1)
    for epsilon in epsilon_arr:
        if method == 'sarsa':
            Q, pi, log = pendulum_sarsa(n_theta, n_thetadot, n_tau, alpha_nom, epsilon, num_episodes, gamma, verbose=False, plot=False)
        elif method == 'q_learning':
            Q, pi, log = pendulum_q_learning(n_theta, n_thetadot, n_tau, alpha_nom, epsilon, num_episodes, gamma, verbose=False, plot=False)
        else:
            raise ValueError(f'Unknown method: {method}')
        plt.plot(np.arange(num_episodes), log['G'], label=fr'$\epsilon$={epsilon:0.1e}')
    plt.xlabel('Episode #')
    plt.ylabel('Return')
    plt.yscale('symlog')
    plt.legend()

    plt.subplot(1, 2, 2)
    for alpha in alpha_arr:
        if method == 'sarsa':
            Q, pi, log = pendulum_sarsa(n_theta, n_thetadot, n_tau, alpha, epsilon_nom, num_episodes, gamma, verbose=False, plot=False)
        elif method == 'q_learning':
            Q, pi, log = pendulum_q_learning(n_theta, n_thetadot, n_tau, alpha, epsilon_nom, num_episodes, gamma, verbose=False, plot=False)
        else:
            raise ValueError(f'Unknown method: {method}')
        plt.plot(np.arange(num_episodes), log['G'], label=fr'$\alpha$={alpha:0.1f}')
    plt.xlabel('Episode #')
    plt.ylabel('Return')
    plt.yscale('symlog')
    plt.legend()
    plt.savefig(f'figures/pendulum/diff_epsilon_alpha_{method}.png', bbox_inches='tight')

    return None

def main():
    verbose = True
    if verbose:
        print('-------------------------- Pendulum --------------------------')
    size = 41
    n_theta = size
    n_thetadot = size
    n_tau = size

    gamma = 0.95
    alpha = 0.3
    epsilon = 0.8
    num_episodes = 3000
    Q, pi, log = pendulum_sarsa(n_theta, n_thetadot, n_tau, alpha, epsilon, num_episodes=num_episodes, gamma=gamma, verbose=verbose, plot=True)
    V, log = pendulum_td0(n_theta, n_thetadot, n_tau, pi, alpha, num_episodes=num_episodes, gamma=gamma, verbose=verbose, method='sarsa', plot=True)
    plot_diff_epsilon_alpha(n_theta, n_thetadot, n_tau, method='sarsa', num_episodes=num_episodes, gamma=gamma)
    
    Q, pi, log = pendulum_q_learning(n_theta, n_thetadot, n_tau, alpha, epsilon, num_episodes=num_episodes, gamma=gamma, verbose=verbose, plot=True)
    V, log = pendulum_td0(n_theta, n_thetadot, n_tau, pi, alpha, num_episodes=num_episodes, gamma=gamma, verbose=verbose, method='q_learning', plot=True)
    plot_diff_epsilon_alpha(n_theta, n_thetadot, n_tau, method='q_learning', num_episodes=num_episodes, gamma=gamma)

if __name__ == '__main__':
    main()
