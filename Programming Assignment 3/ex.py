import numpy as np
import matplotlib.pyplot as plt
import random

class RandomEpisodicTask:
    """
    Generates and holds the random MDP as described in Section 8.6.
    """
    def __init__(self, n_states, n_actions, b, term_prob=0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.b = b
        self.term_prob = term_prob
        
        self.start_state = 0
        self.terminal_state = -1
        
        self.T = {}
        for s in range(self.n_states):
            for a in range(self.n_actions):
                
                possible_next_states = np.random.choice(
                    self.n_states, size=b, replace=False
                )
                
                transitions = []
                prob_next_state = (1.0 - self.term_prob) / self.b
                
                for next_s in possible_next_states:
                    reward = np.random.normal(0, 1)
                    transitions.append((prob_next_state, next_s, reward))
                    
                term_reward = np.random.normal(0, 1)
                transitions.append((self.term_prob, self.terminal_state, term_reward))
                
                self.T[(s, a)] = transitions

    def get_expected_update_components(self, s, a):
        return self.T[(s, a)]

class Planner:
    """
    Implements the planner with Uniform and On-Policy update strategies.
    Uses asynchronous "in-place" expected updates.
    """
    def __init__(self, task, epsilon=0.1, gamma=1.0):
        self.task = task
        self.n_states = task.n_states
        self.n_actions = task.n_actions
        self.total_sa_pairs = self.n_states * self.n_actions
        
        self.epsilon = epsilon # For on-policy simulation
        self.gamma = gamma     # Undiscounted
        
        self.Q = np.zeros((self.n_states, self.n_actions))

    def get_greedy_action(self, s):
        return np.argmax(self.Q[s, :])

    def choose_action_epsilon_greedy(self, s):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return self.get_greedy_action(s)
            
    def policy_evaluation(self, theta=1e-4):
        """
        Computes V(s) for the current *greedy* policy (derived from Q).
        """
        V_eval = np.zeros(self.n_states)
        while True:
            delta = 0
            for s in range(self.n_states):
                v_old = V_eval[s]
                a = self.get_greedy_action(s)
                v_new = 0
                for (prob, next_s, reward) in self.task.get_expected_update_components(s, a):
                    next_v = 0 if next_s == self.task.terminal_state else V_eval[next_s]
                    v_new += prob * (reward + self.gamma * next_v)
                V_eval[s] = v_new
                delta = max(delta, np.abs(v_old - V_eval[s]))
            if delta < theta:
                break
        return V_eval[self.task.start_state]

    def expected_update(self, s, a):
        """
        Performs a single "in-place" expected update on Q(s, a).
        Q(s,a) <- sum[ p(s',r|s,a) * (r + gamma * max_a' Q(s', a')) ]
        """
        q_new = 0
        for (prob, next_s, reward) in self.task.get_expected_update_components(s, a):
            
            if next_s == self.task.terminal_state:
                max_next_q = 0
            else:
                max_next_q = np.max(self.Q[next_s, :]) # Reads from current Q
                
            q_new += prob * (reward + self.gamma * max_next_q)
            
        self.Q[s, a] = q_new # Writes to current Q

    def run_uniform(self, max_backups):
        """
        Runs the Uniform planner (Dyna-Q style).
        'max_backups' is the number of full sweeps.
        """
        results = []
        updates_per_backup = self.total_sa_pairs
        total_updates_to_do = max_backups * updates_per_backup
        
        # Add result at backup 0
        results.append(self.policy_evaluation())

        for update_count in range(total_updates_to_do):
            
            # Sample (s, a) uniformly at random
            s = np.random.randint(self.n_states)
            a = np.random.randint(self.n_actions)
            
            # Perform in-place, asynchronous update
            self.expected_update(s, a)
            
            # Check if we've completed a batch
            if (update_count + 1) % updates_per_backup == 0:
                start_state_value = self.policy_evaluation()
                results.append(start_state_value)
                
        return results

    def run_on_policy(self, max_backups):
        """
        Runs the On-Policy (Trajectory Sampling) planner.
        """
        results = []
        updates_per_backup = self.total_sa_pairs
        total_updates_to_do = max_backups * updates_per_backup
        
        # Add result at backup 0
        results.append(self.policy_evaluation())
        
        s = self.task.start_state
        
        for update_count in range(total_updates_to_do):
            # Simulate one step
            a = self.choose_action_epsilon_greedy(s)
            
            # Update Q(s,a) "in-place"
            self.expected_update(s, a)
            
            # --- Get next state from simulation ---
            transitions = self.task.T[(s, a)]
            probs = [t[0] for t in transitions]
            indices = np.arange(len(transitions))
            
            chosen_idx = np.random.choice(indices, p=probs)
            _prob, next_s, _reward = transitions[chosen_idx]
            
            if next_s == self.task.terminal_state:
                s = self.task.start_state # Reset episode
            else:
                s = next_s # Continue trajectory
                
            # Check if we've completed a batch equivalent to one full backup
            if (update_count + 1) % updates_per_backup == 0:
                start_state_value = self.policy_evaluation()
                results.append(start_state_value)
                
        return results

def run_experiment(n_states, n_actions, b, n_runs, max_backups):
    """
    Runs the full experiment for a given 'b'.
    """
    
    all_uniform_results = []
    all_on_policy_results = []
    
    print(f"\n--- Running Experiment: b={b}, n_states={n_states} ---")
    
    for r in range(n_runs):
        print(f"  Run {r+1}/{n_runs}")
        
        task = RandomEpisodicTask(n_states, n_actions, b)
        
        planner_uniform = Planner(task)
        uniform_res = planner_uniform.run_uniform(max_backups)
        all_uniform_results.append(uniform_res)
        
        planner_on_policy = Planner(task)
        on_policy_res = planner_on_policy.run_on_policy(max_backups)
        all_on_policy_results.append(on_policy_res)
        
    avg_uniform = np.mean(all_uniform_results, axis=0)
    avg_on_policy = np.mean(all_on_policy_results, axis=0)
    
    # --- Plotting with CORRECTED x-axis ---
    updates_per_backup = n_states * n_actions
    x_axis = np.arange(0, max_backups + 1) * updates_per_backup
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, avg_uniform, label='uniform', color='darkgreen', alpha=0.5, linewidth=2)
    plt.plot(x_axis, avg_on_policy, label='on-policy', color='darkgreen', alpha=1.0, linewidth=2)
    
    plt.title(f'Uniform vs. On-Policy Updates (b={b}, {n_states} states)', fontsize=14)
    plt.xlabel('Computation time, in expected updates', fontsize=12)
    plt.ylabel('Value of start state\nunder greedy policy', fontsize=12)
    plt.legend(fontsize=12)
    plt.xlim(0, max_backups * updates_per_backup)
    plt.ylim(0, None)
    plt.grid(True, alpha=0.3)
    
    # Save the figure
    filename = f'figure_8_8_b{b}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {filename}")
    plt.show()
    
    return avg_uniform, avg_on_policy

if __name__ == "__main__":
    
    N_STATES_LOWER = 10000
    N_ACTIONS = 2
    N_RUNS = 5  # Reduced to 5 as requested
    MAX_BACKUPS_LOWER = 10 
    
    # --- Primary Question: Replicate Figure 8.8 (lower part, b=1) ---
    print("=" * 60)
    print("Replicating Figure 8.8 with b=1")
    print("=" * 60)
    avg_uniform_b1, avg_on_policy_b1 = run_experiment(
        N_STATES_LOWER, N_ACTIONS, b=1, n_runs=N_RUNS, max_backups=MAX_BACKUPS_LOWER
    )
    
    # --- Secondary Question: Run with b=3 ---
    print("\n" + "=" * 60)
    print("Running experiment with b=3")
    print("=" * 60)
    avg_uniform_b3, avg_on_policy_b3 = run_experiment(
        N_STATES_LOWER, N_ACTIONS, b=3, n_runs=N_RUNS, max_backups=MAX_BACKUPS_LOWER
    )
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"\nFor b=1:")
    print(f"  Final on-policy value: {avg_on_policy_b1[-1]:.3f}")
    print(f"  Final uniform value:   {avg_uniform_b1[-1]:.3f}")
    print(f"  On-policy advantage:   {avg_on_policy_b1[-1] - avg_uniform_b1[-1]:.3f}")
    
    print(f"\nFor b=3:")
    print(f"  Final on-policy value: {avg_on_policy_b3[-1]:.3f}")
    print(f"  Final uniform value:   {avg_uniform_b3[-1]:.3f}")
    print(f"  On-policy advantage:   {avg_on_policy_b3[-1] - avg_uniform_b3[-1]:.3f}")