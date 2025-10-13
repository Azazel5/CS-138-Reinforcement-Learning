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
        
        # Check for the special case where velocity is (0,0) AND not on the start line
        current_pos_tuple = tuple(self.position)
        if self.velocity == [0, 0] and current_pos_tuple not in self.start_positions:
            # According to the book, velocity components cannot both be zero, except at the start
            # To prevent getting stuck, we can give it a minimal velocity
            # However, a well-trained agent should avoid this state
            # Forcing a minimal velocity might interfere with learning
            # A crash is a more appropriate outcome for this illegal state

            return self.handle_crash()

        # Project the path and check for collisions
        old_pos = self.position
        new_pos = [old_pos[0] - self.velocity[0],
                   old_pos[1] + self.velocity[1]]

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
                # Reward is 0 on finishing
                return tuple(self.position + self.velocity), 0, True

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
        # Reset the car to a random start position with zero velocity
        new_state = self.reset()
        # The episode continues after a crash
        done = False
        # The reward is -1 plus any additional crash penalty
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
                    self.Q[state][action_idx] += alpha * \
                        (G - self.Q[state][action_idx])
                    visited_state_actions.add(state_action)



# --- 3. Experiment and Visualization ---
def generate_trajectory(agent, env, start_pos):
    """
    Generates a single trajectory following the agent's optimal policy.
    Noise is turned off for this demonstration.
    This function now correctly terminates the trajectory if a crash occurs.
    """

    env.position = list(start_pos)
    env.velocity = [0, 0]
    state = tuple(env.position + env.velocity)

    # The trajectory stores (y, x) position tuples
    trajectory = [state[:2]]
    done = False
    
    while not done:
        # Greedily select the best action (no exploration, no noise)
        action_idx = np.argmax(agent.Q[state])
        action = agent.actions[action_idx]

        # Take a step in the environment
        next_state, _, done = env.step(action, noise=False)

        # The car's position after the step
        next_pos = next_state[:2]

        # Check if the car was reset to a starting position due to a crash
        # This is inferred if the new position is a start position, but the old one wasn't
        # This prevents stopping on the very first move if the car stays put
        if next_pos in env.start_positions and state[:2] not in env.start_positions:
            # A crash occurred. We end this trajectory for visualization purposes.
            # We do *not* append the new starting line position

            print("Trajectory crashed, ending visualization.")
            break

        # Update state and record the new position in the trajectory
        state = next_state
        trajectory.append(next_pos)

        # Add a safeguard against infinite loops for undertrained agents
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

    # Create figure and axis
    _, ax = plt.subplots(figsize=(12, 12))

    # Create a custom colormap with distinct colors for each track element
    colors_list = ['black', 'lightgray', 'green', 'red']
    cmap = plt.matplotlib.colors.ListedColormap(colors_list)

    # Display the track with proper color bounds
    im = ax.imshow(grid, cmap=cmap, interpolation='nearest',
                   vmin=-0.5, vmax=3.5)

    # Draw the trajectories
    traj_colors = ['yellow', 'cyan']
    linestyles = ['-', '--']

    for i, (traj_set, label) in enumerate(zip(trajectories, labels)):
        for j, traj in enumerate(traj_set):
            if traj.shape[0] < 2: continue # Skip empty or single-point trajectories
            # Only add a label to the first trajectory of each agent for a clean legend
            legend_label = label if j == 0 else None
            ax.plot(traj[:, 1], traj[:, 0], color=traj_colors[i],
                    linestyle=linestyles[i], linewidth=2.0, label=legend_label,
                    alpha=0.9, marker='o', markersize=3)

    # Add grid lines
    ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
    ax.grid(which='minor', color='gray',
            linestyle='-', linewidth=0.3, alpha=0.5)
    ax.tick_params(which='minor', size=0)
    ax.tick_params(which='major', size=0, labelbottom=False, labelleft=False)

    # Add legend with better positioning
    ax.legend(loc='upper left', fontsize=10)

    # Add title
    ax.set_title(title, fontsize=16, pad=20)

    # Add a color bar to show what each color represents
    boundaries = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = plt.matplotlib.colors.BoundaryNorm(boundaries, cmap.N)
    cbar = plt.colorbar(im, ax=ax, boundaries=boundaries,
                        norm=norm, ticks=[0, 1, 2, 3], shrink=0.7)
    cbar.ax.set_yticklabels(['Wall', 'Track', 'Start', 'Finish'])

    plt.tight_layout()
    plt.show()



if __name__ == '__main__':

    TRACK_A = [
        "#################",
        "####FFFF#########",
        "###OOF###########",
        "##OOOF###########",
        "##OOOF###########",
        "##OOOF###########",
        "##OOOF###########",
        "##OOOF###########",
        "##OOOF###########",
        "##OOOF###########",
        "##OOOF###########",
        "##OOOF###########",
        "##OOOF###########",
        "##OOOOOOOOOOOOO##",
        "##OOOOOOOOOOOOO##",
        "#OOOOOOOOOOOOOO##",
        "#OOOOOOOOOOOOOO##",
        "#OOOOOOOOOOOOOO##",
        "#OOOOOOOOOOOOOO##",
        "#OOOOOOOOOOOOOO##",
        "#OOOOOOOOOOOOOO##",
        "#OOOOOOOOOOOOOO##",
        "#OOOOOOOOOOOOOO##",
        "#OOOOOOOOOOOOOO##",
        "#OOOOOOOOOOOOOO##",
        "##OOOOOOOOOOOOO##",
        "###OOOOOOOOOOOO##",
        "####SSSSSSSSS####",
        "#################",
    ]

    TRACK_A_EDITED = [
        "################################",
        "#SSSSSSOOOOOOOOOOOOOOOOOOOOOO###",
        "#SSSSSSOOOOOOOOOOOOOOOOOOOOOO###",
        "#SSSSSSOOOOOOOOOOOOOOOOO########",
        "##########OOOOOOOOOOOOOOO#######",
        "##########OOOOOOOOOOOOOOO#######",
        "###############OOOOOOOOOOOOOOO##",
        "###############OOOOOOOOOOOOOOO##",
        "############OOOOOOOOOOOOOOOOOO##",
        "############OOOOOOOOOOOOOOOOOO##",
        "######OOOOOOOOOOOOOOOOO#########",
        "######OOOOOOOOOOOOOOOOO#########",
        "###OOOOOOOOOOOOOOOOOOOFFFFFF####",
        "###OOOOOOOOOOOOOOOOOOOFFFFFF####",
        "################################",
    ]

    TRACK_B = [
        "###################################",
        "#############################FFFFF#",
        "#############################OOF###",
        "############################OOOF###",
        "###########################OOOO####",
        "##########################OOOOO####",
        "#########################OOOOOO####",
        "########################OOOOOOO####",
        "#######################OOOOOOOO####",
        "######################OOOOOOOOO####",
        "#####################OOOOOOOOOO####",
        "####################OOOOOOOOOOO####",
        "###################OOOOOOOOOOOO####",
        "#################OOOOOOOOOOOOOO####",
        "###############OOOOOOOOOOOOOOOO####",
        "##############OOOOOOOOOOOOOOOOO####",
        "#############OOOOOOOOOOOOOOOOOO####",
        "############OOOOOOOOOOOOOOOOOOO####",
        "###########OOOOOOOOOOOOOOOOOOOO####",
        "##########OOOOOOOOOOOOOOOOOOOOO####",
        "#########OOOOOOOOOOOOOOOOOOOOOO####",
        "########OOOOOOOOOOOOOOOOOOOOOOO####",
        "#######OOOOOOOOOOOOOOOOOOOOOOOO####",
        "######OOOOOOOOOOOOOOOOOOOOOOOOO####",
        "#####OOOOOOOOOOOOOOOOOOOOOOOOOO####",
        "####OOOOOOOOOOOOOOOOOOOOOOOOOOO####",
        "###OOOOOOOOOOOOOOOOOOOOOOOOOOOO####",
        "##OOOOOOOOOOOOOOOOOOOOOOOOOOOOO####",
        "#OOOOOOOOOOOOOOOOOOOOOOOOOOOOOO####",
        "#SSSSSSSSSSSSSSSSSSSSSSSSSSSSSS####",
        "###################################",
    ]

    TRACK_A_EXACT = [
        "#################",
        "####OOOOOOOOOOOF#",
        "###OOOOOOOOOOOOF#",
        "##OOOOOOOOOOOOOF#",
        "#OOOOOOOOOOOOOOF#",
        "#OOOOOOOOOOOOOOF#",
        "#OOOOOOOOOOOOOOO#",
        "##OOOOOOOOOOOOOO#",
        "##OOOOOOOOOOOOOO#",
        "##OOOOOOOOOO#####", 
        "##OOOOOOOO#######",
        "##OOOOOOOO#######",
        "##OOOOOOOO#######",
        "##OOOOOOOO#######",
        "##OOOOOOOO#######",
        "##OOOOOOOO#######",
        "##OOOOOOOO#######",
        "##OOOOOOOO#######",
        "#OOOOOOOOO#######",
        "#OOOOOOOOO#######",
        "#OOOOOOOOO#######",
        "#OOOOOOOOO#######",
        "#OOOOOOOOO#######",
        "#OOOOOOOOO#######",
        "##OOOOOOOO#######",
        "###OOOOOOO#######",
        "####OOOOOO#######",
        "####SSSSSS#######",
        "#################"
    ]

    TRACK_B_EXACT = [
        "##########OOOOOOOOOOOOOF",
        "###########OOOOOOOOOOOOF",
        "############OOOOOOOOOOOF",
        "#############OOOOOOOOOOF",
        "#############OOOOOOOOOOF",
        "#############OOOOOOOOOOF",
        "#############OOOOOOOOOOF",
        "#############OOOOOOOOOOF",
        "#############OOOOOOOOOOF",
        "############OOOOOOOOOO##",
        "###########OOOOOOOOOO###",
        "##########OOOOOOOOOOO###",
        "#########OOOOOOOOOOOO###",
        "########OOOOOOOOOOOOO###",
        "#######OOOOOOOOOOOOOO###",
        "######OOOOOOOOOOOOOOO###",
        "#####OOOOOOOOOOOOOOOO###",
        "####OOOOOOOOOOOOOOOOO###",
        "###OOOOOOOOOOOOOOOOOO###",
        "##OOOOOOOOOOOOOOOOOOO###",
        "#OOOOOOOOOOOOOOOOOOOO###",
        "OOOOOOOOOOOOOOOOOOOOO###",
        "OOOOOOOOOOOOOOOOOOOOO###",
        "SSSSSSSSSSSSSSSSSSSSS###"
    ]
    
    # You can switch between TRACK_A and TRACK_B here
    TRACK_LAYOUT = TRACK_B_EXACT

    # --- Run Experiment ---
    # Increase episodes for better convergence on more complex tracks
    NUM_EPISODES = 50000

    # 1. Train the "Daredevil" Agent (original problem)
    print("--- Training Daredevil Agent (Standard Penalty) ---")
    daredevil_env = Racetrack(TRACK_LAYOUT, crash_penalty=0)
    daredevil_agent = MCAgent(actions=daredevil_env.actions)
    daredevil_agent.train(daredevil_env, NUM_EPISODES)

    # 2. Train the "Cautious" Agent (additional question)
    print("\n--- Training Cautious Agent (High Crash Penalty) ---")
    cautious_env = Racetrack(TRACK_LAYOUT, crash_penalty=-10) # A penalty of -10 is sufficient
    cautious_agent = MCAgent(actions=cautious_env.actions)
    cautious_agent.train(cautious_env, NUM_EPISODES)

    # --- Generate and Plot Trajectories ---
    # Test from a few different starting positions to see varied paths
    start_positions_to_test = cautious_env.start_positions[::6] # Pick every 6th start pos

    daredevil_trajectories = [generate_trajectory(
        daredevil_agent, daredevil_env, pos) for pos in start_positions_to_test]
    cautious_trajectories = [generate_trajectory(
        cautious_agent, cautious_env, pos) for pos in start_positions_to_test]

    plot_racetrack(
        TRACK_LAYOUT,
        [daredevil_trajectories, cautious_trajectories],
        labels=["Daredevil Policy (Crash Penalty=0)",
                "Cautious Policy (Crash Penalty=-10)"],
        title="Comparison of Optimal Policies under Different Crash Penalties"
    )