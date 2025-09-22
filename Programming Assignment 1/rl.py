import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# --- 1. The Environment ---
class NonstationaryBandit:
    """
    A class for the 10-armed nonstationary bandit problem.
    The true action values (q*) are initialized to zero and then take an
    independent random walk on each step.
    """
    def __init__(self, k=10, walk_mean=0.0, walk_std=0.01, reward_noise_std=1.0):
        self.k = k
        self.walk_mean = walk_mean
        self.walk_std = walk_std
        self.reward_noise_std = reward_noise_std
        self.true_action_values = np.zeros(self.k)

    def step(self, action):
        reward = np.random.normal(loc=self.true_action_values[action], scale=self.reward_noise_std)
        random_increments = np.random.normal(loc=self.walk_mean, scale=self.walk_std, size=self.k)
        self.true_action_values += random_increments
        return reward

    def get_optimal_action(self):
        return np.argmax(self.true_action_values)



# --- 2. The Learning Agent ---
class Agent:
    """
    An epsilon-greedy agent that can use either sample-average or constant
    step-size update rules.
    """
    def __init__(self, k, epsilon, alpha=None):
        self.k = k
        self.epsilon = epsilon
        self.alpha = alpha  # If alpha is None, use sample-average method
        self.q_estimates = np.zeros(k)
        self.action_counts = np.zeros(k, dtype=int)

    def select_action(self):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.k)  # Explore
        else:
            return np.argmax(self.q_estimates) # Exploit

    def update(self, action, reward):
        self.action_counts[action] += 1
        
        if self.alpha is None:
            # Sample-average step size: 1/n
            step_size = 1 / self.action_counts[action]
        else:
            # Constant step size
            step_size = self.alpha
            
        error = reward - self.q_estimates[action]
        self.q_estimates[action] += step_size * error




# --- 3. The Experiment ---
def run_experiment(num_runs=2000, num_steps=10000):
    k = 10
    epsilon = 0.1
    alpha = 0.1
    
    # Store results for each agent type
    results = {
        "Sample Average": {'rewards': np.zeros((num_runs, num_steps)), 'optimal_actions': np.zeros((num_runs, num_steps))},
        "Constant Step-Size": {'rewards': np.zeros((num_runs, num_steps)), 'optimal_actions': np.zeros((num_runs, num_steps))}
    }

    agent_configs = {
        "Sample Average": {"alpha": None},
        "Constant Step-Size": {"alpha": alpha}
    }
    
    for name, config in agent_configs.items():
        print(f"Running agent: {name}")
        for run in tqdm(range(num_runs)):
            bandit = NonstationaryBandit(k=k)
            agent = Agent(k=k, epsilon=epsilon, alpha=config["alpha"])
            
            for step in range(num_steps):
                optimal_action = bandit.get_optimal_action()
                action = agent.select_action()
                reward = bandit.step(action)
                agent.update(action, reward)
                
                # Store data
                results[name]['rewards'][run, step] = reward
                if action == optimal_action:
                    results[name]['optimal_actions'][run, step] = 1
    
    # Average results over all runs
    avg_rewards = {name: np.mean(data['rewards'], axis=0) for name, data in results.items()}
    avg_optimal_actions = {name: np.mean(data['optimal_actions'], axis=0) for name, data in results.items()}

    return avg_rewards, avg_optimal_actions




# --- 4. Plotting the Results ---
if __name__ == '__main__':
    avg_rewards, avg_optimal_actions = run_experiment()

    # Plot Average Reward
    plt.figure(figsize=(12, 6))
    for name, rewards in avg_rewards.items():
        plt.plot(rewards, label=f'{name}')
    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.title("Average Reward on Nonstationary 10-Armed Bandit")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

    # Plot % Optimal Action
    plt.figure(figsize=(12, 6))
    for name, optimal_actions in avg_optimal_actions.items():
        plt.plot(optimal_actions * 100, label=f'{name}') # Convert to percentage
    plt.xlabel("Steps")
    plt.ylabel("% Optimal Action")
    plt.title("% Optimal Action on Nonstationary 10-Armed Bandit")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.gca().set_yticklabels([f'{int(y)}%' for y in plt.gca().get_yticks()])
    plt.show()