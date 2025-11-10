import numpy as np
import random
import pickle
import json, os
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from game import Game

# -----------------------------
# AUXILIARY FUNCTIONS
# -----------------------------

# list of tuples representing all moves in TIC TAC TOE game
ACTIONS = [
    (0, 0), (0, 1), (0, 2),
    (1, 0), (1, 1), (1, 2),
    (2, 0), (2, 1), (2, 2)
]

#function converting board as array to string to represent state in Q table
def board_to_string(board):
    return ''.join(cell if cell is not None else '-' for row in board for cell in row)

# return list of tuples with coordinates of empty cells
def list_possible_moves(board):
    possible_moves = []

    for row, col in ACTIONS:
        if board[row][col] is None:
            possible_moves.append((row, col))
    if not possible_moves:
        return None  # no moves

    return possible_moves

#return tuple with random move
def random_move(board):
    possible_moves = list_possible_moves(board)

    if possible_moves is not None:
        return possible_moves[np.random.randint(len(possible_moves))]
    else:
        return None

def is_board_full(board):
    return all(element is not None for row in board for element in row)


# -----------------------------
# Q-learning PARAMS
# -----------------------------

# Learning pace (learning rate).
# It determines how much new experiences (new Q-values) overwrite old values.
# - High value (e.g. 0.5) → fast learning but unstable.
# - Low value (e.g. 0.001) → slow learning, but more stable.
learning_rate = 0.01

# Factor that discounts future rewards (discount factor).
# It determines how much the agent "cares" about rewards in the future compared to rewards in the here and now.
# - close to 1.0 → the agent learns to plan "for the long term".
# - close to 0 → the agent only looks at immediate benefits.
discount_factor = 0.85

# Exploration rate (using ε).
# Determines how often the agent chooses a random action instead of the best known one.
# - High value (e.g. 0.9) → a lot of testing random moves (exploration).
# - Low value (e.g. 0.1) → mainly using what has already been learned (exploitation).
# exploration_rate parameter is different for each episode and is determined by:
# FORMULA:      ϵ = ϵ_min + (ϵ_max - ϵ_min) * e^(-decay_rate * episode)
# (calculations in the training function)
epsilon_min = 0.01
epsilon_max = 0.9
decay_rate = 0.001


# Number of training episodes (number of times the agent plays the entire game from start to finish).
# The more episodes, the better the agent learns because he has more experience.
# Typically from several thousand to hundreds of thousands.
num_episodes = 500000

# -----------------------------
# CHOOSE MOVE
# -----------------------------
def choose_action(board, Q_table, exploration_rate):
    state = board_to_string(board)

    if random.uniform(0, 1) < exploration_rate or state not in Q_table:
        action = random_move(board)
    else:
        q_values = Q_table[state]
        empty_cells = list_possible_moves(board)
        empty_q_values = [q_values[row, col] for (row, col) in empty_cells]
        max_q_value = max(empty_q_values)
        max_q_indices = [i for i in range(len(empty_cells)) if empty_q_values[i] == max_q_value]
        max_q_index = random.choice(max_q_indices)
        action = empty_cells[max_q_index]

    return action

# -----------------------------
# UPDATE Q
# -----------------------------
def update_q_table(Q_table, state, action, next_state, reward):
    q_values = Q_table.get(state, np.zeros((3, 3))) # Retrieve the Q-values for a particular state from the Q-table dictionary Q.

    # Calculate the maximum Q-value for the next state
    next_q_values = Q_table.get(next_state, np.zeros((3, 3)))
    max_next_q_value = np.max(next_q_values)

    # Q-learning update equation
    q_values[action[0], action[1]] += learning_rate * (reward + discount_factor * max_next_q_value - q_values[action[0], action[1]])

    Q_table[state] = q_values

# -----------------------------
# TRAINING
# -----------------------------
def train_agents(epsilon_min, epsilon_max, decay_rate):
    Q_attack, Q_defence = {}, {}
    game = Game() # creating an instance of the Game class for training

    # Initializing data for charts
    stats = {
        "attack": {"wins": [], "draws": [], "losses": []},
        "defence": {"wins": [], "draws": [], "losses": []}
    }
    episodes_checkpoints = []
    evaluation_games = 1000
    checkpoint_interval = num_episodes // 10

    # Preparing the chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    plt.ion()
    fig.suptitle("Agents learning evaluation vs random enemy")

    # Chart configuration
    axes_config = [
        (ax1, "Offensive agent (X)", "attack"),
        (ax2, "Defensive agent (O)", "defence")
    ]

    colors = {'wins': 'green', 'draws': 'gray', 'losses': 'red'}
    lines = {}

    for ax, title, agent_type in axes_config:
        ax.set_title(title)
        ax.set_xlabel("Episodes")
        ax.set_ylabel("Result ratio (%)", rotation=0, labelpad=40)
        ax.grid(alpha=0.3)
        lines[agent_type] = {}
        for result_type in ['wins', 'draws', 'losses']:
            lines[agent_type][result_type], = ax.plot([], [], label=result_type.capitalize(), color=colors[result_type])
        ax.legend()

    def update_plots():
        """Updates data on charts without a complete redraw"""
        x = np.array(episodes_checkpoints)

        for ax, title, agent_type in axes_config:
            stats_data = stats[agent_type]
            if len(x) == 0 or len(stats_data["wins"]) == 0:
                continue

            # Update line data
            for result_type in ['wins', 'draws', 'losses']:
                y_data = stats_data[result_type]
                min_len = min(len(x), len(y_data))
                lines[agent_type][result_type].set_data(x[:min_len], y_data[:min_len])

            # Adjust axis range
            ax.relim()
            ax.autoscale_view()

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.canvas.draw()
        fig.canvas.flush_events()

    def evaluate_and_record(episode):
        """Performs evaluation and records results"""
        wins_a, draws_a, losses_a = simulate_games("attack", Q_attack, evaluation_games)
        wins_d, draws_d, losses_d = simulate_games("defence", Q_defence, evaluation_games)

        # Record the results of the attack
        stats["attack"]["wins"].append((wins_a / evaluation_games) * 100)
        stats["attack"]["draws"].append((draws_a / evaluation_games) * 100)
        stats["attack"]["losses"].append((losses_a / evaluation_games) * 100)

        # Record the defense results
        stats["defence"]["wins"].append((wins_d / evaluation_games) * 100)
        stats["defence"]["draws"].append((draws_d / evaluation_games) * 100)
        stats["defence"]["losses"].append((losses_d / evaluation_games) * 100)

        episodes_checkpoints.append(episode)

    # Main training loop
    for episode in range(num_episodes):
        last_moves = {"X": None, "O": None}

        # Set exploration rate for this episode
        exploration_rate = epsilon_min + (epsilon_max - epsilon_min) * np.exp(-decay_rate * episode)
        # print(exploration_rate)

        if episode % checkpoint_interval == 0 or episode == num_episodes - 1:
            evaluate_and_record(episode)
            update_plots()
            plt.pause(0.05)

        game.reset_game()

        while not game.game_over:
            Q = Q_attack if game.current_player == 'X' else Q_defence

            # Choose an action
            action = choose_action(game.board, Q, exploration_rate)

            # Save state before move
            state_before = board_to_string(game.board)

            # Make the chosen move
            row, col = action
            game.make_move(row, col)

            # Save state after move
            state_after = board_to_string(game.board)

            # remember the player's last move
            last_moves[game.current_player] = (state_before, action, state_after)


            if game.game_over:
                if game.winner == 'X':
                    update_q_table(Q_attack, *last_moves['X'], reward=1)
                    update_q_table(Q_defence, *last_moves['O'], reward=-1)
                elif game.winner == 'O':
                    update_q_table(Q_attack, *last_moves['X'], reward=-1)
                    update_q_table(Q_defence, *last_moves['O'], reward=1)
                else:  # draw
                    update_q_table(Q_attack, *last_moves['X'], reward=0)
                    update_q_table(Q_defence, *last_moves['O'], reward=1)
            game.switch_player()

    # SMOOTHING ONLY ON WRITE
    def smooth_final_plot():
        """Smooths the graph only for the final output."""

        x_original = np.array(episodes_checkpoints)

        for ax, title, agent_type in axes_config:
            stats_data = stats[agent_type]
            if len(x_original) < 4:  # Not enough points to smooth
                continue

            for result_type in ['wins', 'draws', 'losses']:
                y_original = np.array(stats_data[result_type])

                # Denser points for a smoother graph
                x_smooth = np.linspace(x_original.min(), x_original.max(), 300)

                # We use spline interpolation
                spline = make_interp_spline(x_original, y_original, k=3)
                y_smooth = spline(x_smooth)

                # Update the line with the smoothed data
                lines[agent_type][result_type].set_data(x_smooth, y_smooth)

            ax.relim()
            ax.autoscale_view()


    smooth_final_plot()
    plt.ioff()
    fig.savefig("training_progress.png", dpi=200, bbox_inches="tight")

    # Additionally: save the raw data version too
    fig_raw, (ax1_raw, ax2_raw) = plt.subplots(1, 2, figsize=(12, 5))
    fig_raw.suptitle("Agents learning evaluation vs random enemy (Raw Data)")

    # Plot raw data on a second graph
    for ax, title, agent_type in [(ax1_raw, "Offensive agent (X)", "attack"),
                                  (ax2_raw, "Defensive agent (O)", "defence")]:
        ax.set_title(title)
        ax.set_xlabel("Episodes")
        ax.set_ylabel("Result ratio (%)", rotation=0, labelpad=40)
        ax.grid(alpha=0.3)

        x = np.array(episodes_checkpoints)
        stats_data = stats[agent_type]
        min_len = min(len(x), len(stats_data["wins"]))

        ax.plot(x[:min_len], stats_data["wins"][:min_len], label="Wins", color='green', linewidth=2)
        ax.plot(x[:min_len], stats_data["draws"][:min_len], label="Draws", color='gray', linewidth=2)
        ax.plot(x[:min_len], stats_data["losses"][:min_len], label="Losses", color='red', linewidth=2)
        ax.legend()

    fig_raw.tight_layout(rect=[0, 0, 1, 0.95])
    fig_raw.savefig("training_progress_raw.png", dpi=200, bbox_inches="tight")
    plt.close(fig_raw)

    return Q_attack, Q_defence

# -----------------------------
# SAVE AND LOAD
# -----------------------------
def save_model(Q_table, filename="q_table.pkl"):
    try:
        with open(filename, "wb") as f:
            pickle.dump(Q_table, f)
            print(f"Saved a file named: {filename}")
    except Exception as e:
        print(f"An error occurred while saving: {e}")


def load_model(filename="q_table.pkl"):
    try:
        with open(filename, "rb") as f:
            print(f"File successfully loaded: {filename}!")
            return pickle.load(f)
    except FileNotFoundError:
        print(f"File '{filename}' has not been found.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return None

# -----------------------------
# PERFORM TRAINED MOVE
# -----------------------------
def trained_move(board, Q_table):
    """
        Choosing the best move basen on trained Q-table.

        Args:
            board (np.array): Current state of board.
            Q_table (dict): table with Q values,

        Returns:
            Tuple: (row, col) representing the best move.
            None: if there are no moves available.
        """
    state = board_to_string(board)
    possible_moves = list_possible_moves(board)

    if not possible_moves:
        return None

    if state not in Q_table:
        # print("Unknown state, making a random move.")
        return random_move(board)

    q_values = Q_table[state]

    # Find the Q values for available moves
    empty_q_values = [q_values[row, col] for (row, col) in possible_moves]
    max_q_value = max(empty_q_values)

    # Randomly choose one of all moves with the highest Q value to avoid determinism
    best_moves_indices = [i for i, q_val in enumerate(empty_q_values) if q_val == max_q_value]
    best_move_index = random.choice(best_moves_indices)

    return possible_moves[best_move_index]

# -----------------------------
# EVALUATE MODEL
# -----------------------------
def simulate_games(purpose, model, games=1000):
    wins, draws, losses = 0, 0, 0
    Q_table = model
    game = Game()

    if purpose == 'attack':
        agent_player = 'X'
    elif purpose == 'defence':
        agent_player = 'O'
    else:
        return print("Unknown evaluate purpose, evaluation process breaking...")

    for _ in range(games):
        game.reset_game()
        while not game.game_over:
            if game.current_player == agent_player:  # agent turn
                move = trained_move(game.board, Q_table)
            else:  # random opponent
                possible = list_possible_moves(game.board)
                move = random.choice(possible)

            row, col = move
            game.make_move(row, col)
            game.switch_player()

        if game.winner == agent_player:
            wins += 1
        elif game.winner is None:
            draws += 1
        else:
            losses += 1

    return wins, draws, losses

def evaluate(purpose, model, games=1000):
    wins, draws, losses = simulate_games(purpose, model, games)

    dane = []
    filename = 'effectiveness_vs_parameters.json'

    if os.path.exists(filename):
        with open(filename, 'r') as f:
            try:
                dane = json.load(f)
            except json.JSONDecodeError:
                dane = []

    effectiveness = wins/games*100
    effectiveness_draws = draws/games*100
    loss_ratio = 100 - effectiveness_draws - effectiveness
    stats = {"training_purpose": f"{purpose}", "effectiveness": f"{effectiveness:.2f}%", "effectiveness_draws": f"{effectiveness_draws:.2f}%",
               "loss_ratio":f"{loss_ratio:.2f}%", "wins": wins, "draws": draws, "losses": losses,
               "learning_rate": learning_rate, "discount_factor": discount_factor, "num_episodes": num_episodes,
               "epsilon_min": epsilon_min, "epsilon_max": epsilon_max, "decay_rate": decay_rate}

    dane.append(stats)

    with open(filename, 'w') as f:
        json.dump(dane, f, indent=2)

    print("--------------------------------------------------------------------")
    print(f"PURPOSE: {purpose}")
    print(f"Wins: {wins}, Draws: {draws}, Losses: {losses}")
    print(f"Win-ratio: {effectiveness:.2f}%")
    print(f"Draw-ratio: {effectiveness_draws:.2f}%")
    print(f"Lose-ratio: {loss_ratio:.2f}%")



# RUN MODEL
if __name__ == "__main__":
    print("Loading trained model...")
    Qa = load_model("q_table_A.pkl")
    Qd = load_model("q_table_D.pkl")
    if (Qa or Qd) is None:
        print("Agents Training...")
        Qa, Qd = train_agents(epsilon_min, epsilon_max, decay_rate)
        save_model(Qa, "q_table_A.pkl")
        save_model(Qd, "q_table_D.pkl")
        evaluate("attack", Qa, 10000)
        evaluate("defence", Qd, 10000)
    evaluate("attack", Qa, 10000)
    evaluate("defence", Qd, 10000)







