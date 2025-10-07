import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt


# --- 1. The Racetrack Environment ---
class Racetrack:
    """
    Implements the Racetrack environment.

    The state is represented as (y, x, vy, vx).
    Actions are tuples (dv_y, dv_x) where each component is in {-1, 0, 1}.
    """

    def __init__(self, track_layout, crash_penalty=0):
        # The track layout is a list of strings representing the grid
        self.track_layout = track_layout

        # The penalty for crashing into a wall
        self.crash_penalty = crash_penalty

        # Parse the layout to create a NumPy grid and find key locations
        self.parse_layout()

        # Define the 9 possible actions (3x3 grid of velocity changes)
        self.actions = [(dv_y, dv_x) for dv_y in [-1, 0, 1]
                         for dv_x in [-1, 0, 1]]

    def parse_layout(self):
        # Convert the string layout to a numerical grid
        # 0: Wall, 1: Track, 2: Start, 3: Finish
        self.track = np.array([list(row) for row in self.track_layout])
        track_map = {'#': 0, 'O': 1, 'S': 2, 'F': 3}
        self.grid = np.vectorize(track_map.get)(self.track)

        # Get the coordinates of start and finish lines.
        self.start_positions = list(zip(*np.where(self.grid == 2)))
        self.finish_line_positions = list(zip(*np.where(self.grid == 3)))

    def reset(self):
        # Place the car at a random starting position with zero velocity
        start_pos = random.choice(self.start_positions)
        self.position = list(start_pos)
        self.velocity = [0, 0]
        return tuple(self.position + self.velocity)

    def step(self, action, noise=True):
        # With 0.1 probability, the action fails and velocity increments are zero
        if noise and random.random() < 0.1:
            action = (0, 0)

        # Update velocity based on the action.
        self.velocity[0] += action[0]
        self.velocity[1] += action[1]

        # Clamp velocity components to be between 0 and 4 (inclusive)
        self.velocity[0] = max(0, min(4, self.velocity[0]))
        self.velocity[1] = max(0, min(4, self.velocity[1]))

        # Check for the special case where velocity is (0,0) AND not pon the start line
        # This is not allowed as per the problem description
        current_pos_tuple = tuple(self.position)
        if self.velocity == [0, 0] and current_pos_tuple not in self.start_positions:
            # This action is illegal, so we revert to a minimal forward velocity

            self.velocity = [0, 1]


        # Project the path and check for collisions
        old_pos = self.position
        new_pos = [old_pos[0] - self.velocity[0], old_pos[1] + self.velocity[1]]

        # The path is a line from old_pos to new_pos. We check all grid cells
        # on this line for collisions
        num_points = int(np.max(np.abs(np.subtract(new_pos, old_pos)))) + 1
        path_y = np.linspace(old_pos[0], new_pos[0], num_points, dtype=int)
        path_x = np.linspace(old_pos[1], new_pos[1], num_points, dtype=int)

        for y, x in zip(path_y, path_x):
            # Check if the car has gone out of bounds
            if not (0 <= y < self.grid.shape[0] and 0 <= x < self.grid.shape[1]):
                return self.handle_crash()

            # Check if the car hits the finish line
            if self.grid[y, x] == 3:
                self.position = [y, x]
                return tuple(self.position + self.velocity), 0, True # Reward is 0 on finishing

            # Check if the car hits a wall.
            if self.grid[y, x] == 0:
                return self.handle_crash()

        # If no collision, update the car's position
        self.position = new_pos
        state = tuple(self.position + self.velocity)

        # Standard penalty for each step
        reward = -1 
        done = False
        return state, reward, done

    def handle_crash(self):
        # Reset the car to a random start position with zero velocity.
        new_state = self.reset()
        # The episode continues after a crash.
        done = False
        # The reward is -1 plus any additional crash penalty.
        reward = -1 + self.crash_penalty
        return new_state, reward, done

# --- 2. The Monte Carlo Agent ---

class MCAgent:
    """
    Implements an On-Policy First-Visit Monte Carlo Control agent with an
    epsilon-greedy policy.
    """
    def __init__(self, actions, epsilon=0.1, gamma=1.0):
        self.actions = actions
        self.epsilon = epsilon
        self.gamma = gamma 

        # The agent's "brain"
        # Q(s,a): Expected return for state-action pair
        self.Q = defaultdict(lambda: np.zeros(len(self.actions)))

        # N(s,a): Count of visits to a state-action pair for averaging
        self.N = defaultdict(lambda: np.zeros(len(self.actions), dtype=int))

    def get_action(self, state):
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            # Explore: choose a random action
            return random.choice(range(len(self.actions)))
        else:
            # Exploit: choose the best known action
            return np.argmax(self.Q[state])

    def train(self, env, num_episodes, max_steps_per_episode=1000):
        # Loop over a number of episodes to learn
        for _ in tqdm(range(num_episodes), desc="Training Agent"):
            # Generate one full episode following the current policy
            episode = []
            state = env.reset()
            done = False
            # while not done:
            #     action_idx = self.get_action(state)
            #     action = self.actions[action_idx]
            #     next_state, reward, done = env.step(action)
            #     episode.append((state, action_idx, reward))
            #     state = next_state

            for _ in range(max_steps_per_episode):
                action_idx = self.get_action(state)
                action = self.actions[action_idx]
                next_state, reward, done = env.step(action)
                episode.append((state, action_idx, reward))
                state = next_state
                if done:
                    break

            # Learn from the episode using first-visit Monte Carlo
            G = 0 
            visited_state_actions = set()

            # Iterate backward through the episode's history
            for state, action_idx, reward in reversed(episode):
                G = self.gamma * G + reward
                state_action = (state, action_idx)

                # If this is the first time we've seen this state-action pair in
                # this episode, update our Q-value estimate
                if state_action not in visited_state_actions:
                    self.N[state][action_idx] += 1
                    # Update Q by taking the average of all returns seen so far
                    alpha = 1 / self.N[state][action_idx]
                    self.Q[state][action_idx] += alpha * (G - self.Q[state][action_idx])
                    visited_state_actions.add(state_action)

# --- 3. Experiment and Visualization ---

def generate_trajectory(agent, env, start_pos):
    """
    Generates a single trajectory following the agent's optimal policy.
    Noise is turned off for this demonstration.
    """
    env.position = list(start_pos)
    env.velocity = [0, 0]
    state = tuple(env.position + env.velocity)

    trajectory = [env.position]
    done = False
    while not done:
        # Greedily select the best action (no exploration, no noise)
        action_idx = np.argmax(agent.Q[state])
        action = agent.actions[action_idx]
        state, _, done = env.step(action, noise=False)
        trajectory.append(env.position)

        # Add a safeguard against infinite loops for undertrained agents.
        if len(trajectory) > 200:
            print("Trajectory too long, breaking.")
            break

    return np.array(trajectory)

def plot_racetrack(track_layout, trajectories, labels, title):
    """
    Visualizes the racetrack and the optimal trajectories
    """
    track = np.array([list(row) for row in track_layout])
    track_map = {'#': 0, 'O': 1, 'S': 2, 'F': 3}
    grid = np.vectorize(track_map.get)(track)

    # Create a colormap for visualization.
    cmap = plt.cm.get_cmap('bone_r', 4)

    plt.figure(figsize=(10, 10))
    plt.imshow(grid, cmap=cmap, interpolation='nearest')

    # Draw the trajectories
    colors = ['r', 'b']
    linestyles = ['-', '--']
    for i, (traj_set, label) in enumerate(zip(trajectories, labels)):
        for j, traj in enumerate(traj_set):
            # Only add a label to the first trajectory of each agent for a clean legend
            legend_label = label if j == 0 else None
            plt.plot(traj[:, 1], traj[:, 0], color=colors[i], linestyle=linestyles[i],
                     linewidth=2, label=legend_label)

    # Add grid, legend, and title for a professional-looking plot
    ax = plt.gca()
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=0.5, alpha=0.3)
    ax.set_xticks(np.arange(-.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, grid.shape[0], 1), minor=True)
    ax.grid(which='minor', axis='both', linestyle='-', color='k', linewidth=0.5, alpha=0.1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.legend()
    plt.title(title, fontsize=16)
    plt.show()


if __name__ == '__main__':

    TRACK_LAYOUT_B = [
        "##################################",
        "#############################FF###",
        "############################F#####",
        "###########################F######",
        "##########################F#######",
        "#########################F########",
        "########################F#########",
        "#######################F##########",
        "######################F###########",
        "#####################F############",
        "####################F#############",
        "###################F##############",
        "##################F###############",
        "#################F################",
        "################F#################",
        "###############O##################",
        "##############OO##################",
        "#############OOO##################",
        "############OOOO##################",
        "###########OOOOO##################",
        "##########OOOOOO##################",
        "#########OOOOOOO##################",
        "########OOOOOOOO##################",
        "#######OOOOOOOOO##################",
        "######OOOOOOOOOO##################",
        "#####OOOOOOOOOOO##################",
        "####OOOOOOOOOOOO##################",
        "###OOOOOOOOOOOOO##################",
        "##OOOOOOOOOOOOOO##################",
        "#OOOOOOOOOOOOOOO##################",
        "#OOOOOOOOOOOOOOO##################",
        "OOOOOOOOOOOOOOOO##################",
        "SSSSSSSSSSSSSSSS##################",
    ]

    # --- Run Experiment ---
    NUM_EPISODES = 2000

    # 1. Train the "Daredevil" Agent (original problem)
    print("--- Training Daredevil Agent (Standard Penalty) ---")
    daredevil_env = Racetrack(TRACK_LAYOUT_B, crash_penalty=0)
    daredevil_agent = MCAgent(actions=daredevil_env.actions)
    daredevil_agent.train(daredevil_env, NUM_EPISODES)

    # 2. Train the "Cautious" Agent (additional question)
    print("\n--- Training Cautious Agent (High Crash Penalty) ---")
    cautious_env = Racetrack(TRACK_LAYOUT_B, crash_penalty=-100)
    cautious_agent = MCAgent(actions=cautious_env.actions)
    cautious_agent.train(cautious_env, NUM_EPISODES)

    # --- Generate and Plot Trajectories ---
    start_positions_to_test = cautious_env.start_positions[::4]

    daredevil_trajectories = [generate_trajectory(daredevil_agent, daredevil_env, pos) for pos in start_positions_to_test]
    cautious_trajectories = [generate_trajectory(cautious_agent, cautious_env, pos) for pos in start_positions_to_test]

    plot_racetrack(
        TRACK_LAYOUT_B,
        [daredevil_trajectories, cautious_trajectories],
        labels=["Daredevil Policy (Crash Penalty=0)", "Cautious Policy (Crash Penalty=-100)"],
        title="Comparison of Optimal Policies under Different Crash Penalties"
    )