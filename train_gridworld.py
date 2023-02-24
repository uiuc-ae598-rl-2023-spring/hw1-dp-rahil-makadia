import random
import numpy as np
import matplotlib.pyplot as plt
import gridworld

def gridworld_policy_iteration(hard_version, theta=1e-16, gamma=0.95, verbose=False):
    if verbose:
        print('------------- Policy Iteration -------------')
        print(f'hard_version = {hard_version}, theta = {theta}, gamma = {gamma}')
    # Create environment
    env = gridworld.GridWorld(hard_version)
    # Initialize simulation
    s = env.reset()

    # apply policy iteration to find optimal policy and value function
    # initialize value function
    V = np.zeros(env.num_states)
    # initialize policy
    pi = 0*np.ones(env.num_states, dtype=int)

    log = {'n_iter': [], 'V': []}
    while True:
        # Step 2: policy evaluation
        delta = np.inf
        n_iter = 0
        while delta >= theta:
            n_iter += 1
            delta = 0
            for s in range(env.num_states):
                v = V[s]
                V[s] = 0
                for s1 in range(env.num_states):
                    V[s] += env.p(s1, s, pi[s]) * (env.r(s, pi[s]) + gamma * V[s1])
                # print(f'V[{s}] = {V[s]}')
                delta = max(delta, abs(v - V[s]))
                # print(f'delta = {delta}')
        if verbose:
            print(f'n_iter = {n_iter}')
        log['n_iter'].append(n_iter)
        log['V'].append(V.copy())

        # Step 3: policy improvement
        policy_stable = True
        for s in range(env.num_states):
            old_action = pi[s]
            piTemp = np.zeros(env.num_actions)
            for s1 in range(env.num_states):
                for a in range(env.num_actions):
                    piTemp[a] += env.p(s1, s, a) * (env.r(s, a) + gamma * V[s1])
            pi[s] = np.argmax(piTemp)
            if old_action != pi[s]:
                # print(f'old_action = {old_action}, pi[s] = {pi[s]}, diff = {old_action - pi[s]}')
                policy_stable = False
        if policy_stable:
            if verbose:
                print('policy is stable!')
            break
    if verbose:
        print(f'V: \n {V.reshape(5,5)}')
        print(f'pi: \n {pi.reshape(5,5)}')
    
    # Create environment
    env = gridworld.GridWorld(hard_version)
    # Initialize simulation
    s = env.reset()

    # Create log to store data from simulation
    log['t'] = [0]
    log['s'] = [s]
    log['a'] = []
    log['r'] = []

    # Simulate until episode is done
    done = False
    while not done:
        a = pi[s]
        s, r, done = env.step(a)
        log['t'].append(log['t'][-1] + 1)
        log['s'].append(s)
        log['a'].append(a)
        log['r'].append(r)
    # Plot data and save to png file
    plt.figure(figsize=(10, 5), dpi=100)
    plt.subplot(1, 2, 1)
    plt.plot(log['t'], log['s'], label='s')
    plt.plot(log['t'][:-1], log['a'], label='a')
    plt.plot(log['t'][:-1], log['r'], label='r')
    plt.xlabel('t')
    plt.ylabel('State / Action / Reward')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(log['n_iter'], [np.mean(V) for V in log['V']], label='V')
    plt.xlabel('Number of iterations')
    plt.ylabel('mean(V)')
    plt.savefig('figures/gridworld/train_gridworld_policy_iteration.png')
    return V, pi, log

def gridworld_value_iteration(hard_version, theta=1e-16, gamma=0.95, verbose=False):  # sourcery skip: sum-comprehension
    if verbose:
        print('------------- Value Iteration -------------')
        print(f'hard_version = {hard_version}, theta = {theta}, gamma = {gamma}')
    # Create environment
    env = gridworld.GridWorld(hard_version)
    # Initialize simulation
    s = env.reset()

    # apply value iteration to find optimal policy and value function
    # initialize value function
    V = np.zeros(env.num_states)
    # initialize policy
    pi = 0*np.ones(env.num_states, dtype=int)

    log = {'n_iter': [], 'V': []}
    n_iter = 0
    delta = np.inf
    while delta >= theta:
        n_iter += 1
        delta = 0
        for s in range(env.num_states):
            v = V[s]
            V[s] = 0
            for a in range(env.num_actions):
                VTemp = 0
                for s1 in range(env.num_states):
                    VTemp += env.p(s1, s, a) * (env.r(s, a) + gamma * V[s1])
                V[s] = max(V[s], VTemp)
            delta = max(delta, abs(v - V[s]))
        log['n_iter'].append(n_iter)
        log['V'].append(V.copy())
    if verbose:
        print(f'n_iter = {n_iter}')
        print(f'V: \n {V.reshape(5,5)}')

    # Step 3: policy improvement (deterministic this time)
    for s in range(env.num_states):
        piTemp = np.zeros(env.num_actions)
        for a in range(env.num_actions):
            for s1 in range(env.num_states):
                piTemp[a] += env.p(s1, s, a) * (env.r(s, a) + gamma * V[s1])
        pi[s] = np.argmax(piTemp)
    if verbose:
        print(f'pi: \n {pi.reshape(5,5)}')

    # Create environment
    env = gridworld.GridWorld(hard_version)
    # Initialize simulation
    s = env.reset()

    # Create log to store data from simulation
    log['t'] = [0]
    log['s'] = [s]
    log['a'] = []
    log['r'] = []

    # Simulate until episode is done
    done = False
    while not done:
        a = pi[s]
        s, r, done = env.step(a)
        log['t'].append(log['t'][-1] + 1)
        log['s'].append(s)
        log['a'].append(a)
        log['r'].append(r)
    # Plot data and save to png file
    plt.figure(figsize=(10, 5), dpi=100)
    plt.subplot(1, 2, 1)
    plt.plot(log['t'], log['s'], label='s')
    plt.plot(log['t'][:-1], log['a'], label='a')
    plt.plot(log['t'][:-1], log['r'], label='r')
    plt.xlabel('t')
    plt.ylabel('State / Action / Reward')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(log['n_iter'], [np.mean(V) for V in log['V']], label='V')
    plt.xlabel('Number of iterations')
    plt.ylabel('mean(V)')
    plt.savefig('figures/gridworld/train_gridworld_value_iteration.png')
    return V, pi, log


def epsilon_greedy_action(Q, epsilon, s, return_best_action=False):
    # initialize policy
    pi = np.ones(Q.shape[1]) * epsilon / Q.shape[1]
    # Choose A from S using policy derived from Q (e.g., "-greedy)
    a_star = np.argmax(Q[s])
    # add probability of 1-epsilon to greedy action
    pi[a_star] += 1 - epsilon
    # choose action according to pi
    a = np.random.choice(Q.shape[1], p=pi)
    return a_star if return_best_action else a

def gridworld_sarsa(hard_version, alpha, epsilon, num_episodes=100, gamma=0.95, verbose=False):
    # sourcery skip: extract-duplicate-method
    if verbose:
        print('------------- SARSA -------------')
        print(f'hard_version = {hard_version}, alpha = {alpha}, epsilon = {epsilon}, num_episodes = {num_episodes}, gamma = {gamma}')
    # Create environment
    env = gridworld.GridWorld(hard_version)

    # apply SARSA to find optimal policy and value function
    # initialize value function
    Q = np.zeros((env.num_states, env.num_actions))
    log = {'t': [], 's': [], 'a': [], 'r': [], 'G': []}
    for _ in range(num_episodes):
        # Initialize simulation
        s = env.reset()
        log['t'] = [0]
        log['s'] = [s]
        log['a'] = []
        log['r'] = []
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
            G += r*gamma**n_iter
            n_iter += 1
        log['G'].append(G)
    pi = np.argmax(Q, axis=1)
    if verbose:
        print(f'Q: \n {Q}')
        print(f'pi: \n {pi.reshape(5,5)}')
    # Plot data and save to png file
    plt.figure(figsize=(10, 5), dpi=100)
    plt.subplot(1, 2, 1)
    plt.plot(log['t'], log['s'], label='s')
    plt.plot(log['t'][:-1], log['a'], label='a')
    plt.plot(log['t'][:-1], log['r'], label='r')
    plt.xlabel('t')
    plt.ylabel('State / Action / Reward')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(np.arange(num_episodes), log['G'], '-', label='G')
    plt.xlabel('Number of episodes')
    plt.ylabel('Return')
    plt.legend()
    plt.savefig('figures/gridworld/train_gridworld_sarsa.png')
    return Q, pi, log

def gridworld_q_learning(hard_version, alpha, epsilon, num_episodes=100, gamma=0.95, verbose=False):
    # sourcery skip: extract-duplicate-method
    if verbose:
        print('------------- Q-Learning -------------')
        print(f'hard_version = {hard_version}, alpha = {alpha}, epsilon = {epsilon}, num_episodes = {num_episodes}, gamma = {gamma}')
    # Create environment
    env = gridworld.GridWorld(hard_version)

    # apply Q-Learning to find optimal policy and value function
    # initialize value function
    Q = np.zeros((env.num_states, env.num_actions))
    log = {'t': [], 's': [], 'a': [], 'r': [], 'G': []}
    for _ in range(num_episodes):
        # Initialize simulation
        s = env.reset()
        log['t'] = [0]
        log['s'] = [s]
        log['a'] = []
        log['r'] = []
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
            G += r*gamma**n_iter
            n_iter += 1
        log['G'].append(G)
    pi = np.argmax(Q, axis=1)
    if verbose:
        print(f'Q: \n {Q}')
        print(f'pi: \n {pi.reshape(5,5)}')
    # Plot data and save to png file
    plt.figure(figsize=(10, 5), dpi=100)
    plt.subplot(1, 2, 1)
    plt.plot(log['t'], log['s'], label='s')
    plt.plot(log['t'][:-1], log['a'], label='a')
    plt.plot(log['t'][:-1], log['r'], label='r')
    plt.xlabel('t')
    plt.ylabel('State / Action / Reward')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(np.arange(num_episodes), log['G'], '-', label='G')
    plt.xlabel('Number of episodes')
    plt.ylabel('Return')
    plt.legend()
    plt.savefig('figures/gridworld/train_gridworld_q_learning.png')
    return Q, pi, log

def gridworld_td0(hard_version, pi, alpha, num_episodes=100, gamma=0.95, verbose=False, suffix=''):
    # sourcery skip: extract-duplicate-method
    if verbose:
        print('------------- TD(0) -------------')
        print(f'hard_version = {hard_version}, alpha = {alpha}, num_episodes = {num_episodes}, gamma = {gamma}')
    # Create environment
    env = gridworld.GridWorld(hard_version)

    # apply TD(0) to find optimal value function
    # initialize value function
    V = np.zeros(env.num_states)
    log = {'t': [], 's': [], 'r': [], 'G': []}
    for _ in range(num_episodes):
        # Initialize simulation
        s = env.reset()
        log['t'] = [0]
        log['s'] = [s]
        log['a'] = []
        log['r'] = []
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
            G += r*gamma**n_iter
            n_iter += 1
        log['G'].append(G)
    if verbose:
        print(f'V: \n {V}')
    # Plot data and save to png file
    plt.figure(figsize=(10, 5), dpi=100)
    plt.subplot(1, 2, 1)
    plt.plot(log['t'], log['s'], label='s')
    plt.plot(log['t'][:-1], log['a'], label='a')
    plt.plot(log['t'][:-1], log['r'], label='r')
    plt.xlabel('t')
    plt.ylabel('State / Action / Reward')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(np.arange(env.num_states), V, '-', label='V')
    plt.xlabel('Number of episodes')
    plt.ylabel('State-value function')
    plt.legend()
    plt.savefig(f'figures/gridworld/train_gridworld_td0{suffix}.png')
    return V, log

def main():
    verbose = True
    hard_version = False
    theta = 1e-16
    gamma = 0.95
    V, pi, log = gridworld_policy_iteration(hard_version, theta, gamma, verbose)
    V, pi, log = gridworld_value_iteration(hard_version, theta, gamma, verbose)

    alpha = 0.5
    epsilon = 0.1
    num_episodes = 1000
    Q, pi, log = gridworld_sarsa(hard_version, alpha, epsilon, num_episodes, gamma, verbose)
    V, log = gridworld_td0(hard_version, pi, alpha, num_episodes, gamma, verbose, suffix='_sarsa')
    Q, pi, log = gridworld_q_learning(hard_version, alpha, epsilon, num_episodes, gamma, verbose)
    V, log = gridworld_td0(hard_version, pi, alpha, num_episodes, gamma, verbose, suffix='_q_learning')

if __name__ == '__main__':
    main()
