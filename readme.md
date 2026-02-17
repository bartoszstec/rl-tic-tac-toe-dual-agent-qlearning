# Tic-Tac-Toe: Dual-Agent Reinforcement Learning

This project re-imagines the classic Tic-Tac-Toe experience by leveraging **Reinforcement Learning (RL)** to power all PvE and EvP modes. It features a complete set of game modes: Player vs. Player, Player vs. AI, and AI vs. Player.

The core technical distinction is the deployment of **two specialized Q-Learning models**: a high-win-rate **Offensive Agent** and a risk-minimizing **Defensive Agent**. This dual-agent architecture demonstrates advanced proficiency in custom RL training, model specialization, and strategic policy design.

## 🎮 Try it Out 
See demo: [DEMO](https://tic-tac-toe-cafkcdgbbrbzcnbt.northeurope-01.azurewebsites.net)

## ✨ Functionalities
- **PvP** Mode – classic two-player game.  
- **PVE** Mode – playing against the agent (you: attack, agent: defends).  
- **EVP** Mode – playing against the agent (you: defend, agent: attacks).
- Two RL agents:
  - **offensive** – his main target is to win the game as quickly as possible,  
  - **defensive** – his target is mainly to draw the game, but also values victory when you make mistakes.  
- Game history page.
- Simple web interface.  

## 🤖 Reinforcement Learning Agents
The project utilizes a simple **Q-Learning model** for agent training. Crucially, instead of a single universal agent, it employs **two independently trained Q-Learning models**, each optimized for a distinct strategic objective. The high performance achieved is a direct result of tailored training and meticulous design choices.

### 🏆 Agent Performance Metrics

The agents' effectiveness was rigorously evaluated by simulating **10,000 games** against a **random opponent**.

|        Agent        |     Strategy Goal     | Win Ratio  | Draw Ratio | Loss Ratio |
|:---:|:---:|:---:| :---: | :---: |
| **Attacking Agent** |   Maximize Victory    | **89.24%** | 8.36% | 2.40% |
| **Defensive Agent** | Minimize Failure Risk |   63.54%   | **28.00%** | 8.46% |

### Why Agents Stand Out?

1. Strategic Reward Shaping:

The difference in performance stems from implementing a distinct Reward System for each agent, effectively shaping their strategy:
- Attacking Agent (Player 'X'): Adopts an aggressive strategy. It is awarded +1 for a win, 0 for a draw, and a penalty of −1 for a loss. To incentivize rapid offense, it receives a small negative reward (punishment) for every non-terminal step (a small penalty for prolonging the game).
- Defensive Agent (Player 'O'): Adopts a conservative strategy focused on resilience. It receives +1 for a win, +1 for a draw, and −1 for a loss. The agent is not penalized for the number of steps. A draw is weighted equally to a win because, due to the inherent first-move advantage in Tic-Tac-Toe, achieving a draw is often the optimal outcome for the second player ('O') against an opponent playing without misplay.

2. Exponential Exploration Decay:

An Exponential Decay schedule is used for the exploration rate (ϵ), which optimizes the trade-off between exploring new moves and exploiting learned knowledge.

The exploration rate for each episode is calculated using the formula:

`ϵ = ϵ_min + (ϵ_max - ϵ_min) * e^(-decay_rate * episode)`

This approach allows the agent to begin with high randomness (exploration) and then quickly and smoothly transition to exploitation (selecting the best moves), leading to faster and more stable convergence compared to linear decay methods.

3. Random Tie-Breaking for Action Selection:

When the Q-table presents multiple moves with the same highest Q-value, the agent randomly selects one of them. This technique, known as **Random Tie-Breaking**, is crucial because it:
- Prevents Determinism: It avoids the agent falling into predictable, suboptimal, or cyclical move patterns.
- Encourages Subtle Exploration: It supports continuous, low-level testing of the best possible action paths, refining the strategy even after the main learning phase is complete.

### 📈 Learning Vizualization
This chart illustrates the performance improvement of the learning agents throughout the training process. The metric is based on periodic evaluations—conducted every few thousand training episodes—where the agents competed against a randomized opponent strategy. The trend line demonstrates the agents' increasing proficiency over time.

Smoothed Performance:
![Learning Graph](training_progress_best.png)

Raw Training Data:
![Learning Graph](training_progress_raw_best.png)

## 🛠️ Project structure
- **Backend (Python):**  
  - `app.py` – server startup and communication logic.  
  - `game.py` – game rules and board state management.  
  - `model.py` – implementation of RL and Q-learning agents.

Backend is implemented to handle multiple games by generating unique ID for every game and store them in server's dict. **Modular approach allows easy scalability**.

- **Frontend (HTML/JS/CSS):**  
  - `index.html` + `styles.css` – UI and game design.  
  - `game.js`, `main.js` – interface logic and player interactions.  
  - `history.html` – viewing the history of played games.

Simple and easy to use interface.


## 🚀 Starting instructions
### ✅ Prerequisites
- Python 3.12+ 
- pip  25.2


1. Clone repository.
```bash
   git clone https://github.com/bartoszstec/tic_tac_toe
```
1. Create virtual environment and activate it.
```bash
  python -m venv venv
  source venv/bin/activate   # on Linux / macOS
  venv\Scripts\activate      # on Windows
```
3. Install dependencies (`requirements.txt`).
```bash
   pip install -r requirements.txt
```
4. Run model.py to train agents
```bash
python model.py
```
5. Run Flask server.
 - Option 1: run directly:
```bash
   python app.py
```

 - Option 2: using Flask CLI:
```bash
   flask run
```

6. Go on flask server link and play
- By default, Flask runs on:
👉 http://127.0.0.1:5000

## 📌 Possible development direction
- Interface improvements.
- Online play option.
- Introduction of more advanced ML agents.


## 📄 License
This project is licensed under the MIT License.
See the [LICENSE](./LICENSE) file for details.
