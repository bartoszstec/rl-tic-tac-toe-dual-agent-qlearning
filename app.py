from flask import Flask, request, jsonify, session
from flask import render_template
import json, os
from model import trained_move, load_model
from game import Game
import uuid

app = Flask(__name__)
app.secret_key = 'BAD_SECRET_KEY'

games = {}  # Main server dictionary with game state keys and values,
            # dictionary: key = session_id, value = Game object

Q_ATTACK = load_model("q_table_A_best.pkl")      # loading Q_table to offensive strategy
Q_DEFENCE = load_model("q_table_D_best.pkl")     # loading Q_table to defensive strategy

def generate_unique_id():
    return str(uuid.uuid4()) # generates a random unique UUID version 4

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/start-game')
def start_game():
    session_id = generate_unique_id()   # function for generating ID (e.g. uuid4)
    session['session_id'] = session_id
    games[session_id] = Game()          # create a new board for the user and storing Game object in dictionary
    return jsonify({"status": "a unique board was created for the user"})

@app.route('/change-strategy', methods = ['POST'])
def change_strategy():
    # Get unique game ID for this session
    session_id = session.get('session_id')
    if not session_id:
        return jsonify({"error": "No session!"}), 400

    # storing the strategy from the frontend
    data = request.get_json()
    strategy = data["strategy"]

    # assigning an instance of a game object to the game variable and changing the strategy value for a given instance
    game = games[session_id]
    game.strategy = strategy

    return jsonify({"status": "Strategy updated"}), 200

@app.route('/history')
def history():
    filename = 'games_history.json'
    games_history = []

    if os.path.exists(filename):
        with open(filename, 'r') as f:
            try:
                games_history = json.load(f)
            except json.JSONDecodeError:
                games_history = []

    return render_template('history.html', games_history = games_history)

@app.route('/AI-move', methods = ['POST'])
def ai_move():
    # Get unique game ID for this session
    session_id = session.get('session_id')
    if not session_id:
        return jsonify({"error": "No session!"}), 400

    # assigning an instance of a game object to the game variable
    game = games[session_id]

    # assigning proper Q_table based on chosen strategy
    if game.strategy == "attack":
        Q_table = Q_ATTACK
    elif game.strategy == "defence":
        Q_table = Q_DEFENCE
    else:
        return jsonify({"error": "Unknown strategy"}), 400

    current_player = game.current_player  # storing current_player, this is player that is going to move
    board = game.board                      # storing current board to variable
    move = trained_move(board, Q_table)     # AI agent choosing the best possible move

    row, col = move                 # storing move to row and col variables

    try:
        game.make_move(row, col)    # make move is main game class method that involves checking if square(represented as row, col) is taken by any player,
                                    # saving done move to game's board, manages game value properties to represent state of game through values as:
                                    # (game_over, winner, winning_line, current_player)

    except ValueError as e:
        # Return error statement
        return jsonify({"error": str(e)}), 400

    winner = game.winner                # default = None, changing to current_player in case of victory
    winning_line = game.winning_line    # default = None, coordinates of winning line
    draw = False                        # a flag indicating a tie

    if game.check_full_board():             # checking is game draw/over or has to be continued
        draw = True
    if game.game_over:
        #update_q_on_game_end()
        # wywołanie metody w modelu która zaktualizuje stan Q-tabeli odpowiedniego agenta przy końcu gry
        game.reset_game()
    else:
        #update_q_on_move()
        # wywołanie metody w modelu która zaktualizuje stan Q-tabeli o nagrody ongoing
        game.switch_player()    # method switching current_player from X to O and from O to X, switched player is the next player's move
                                # WARNING: has to be executed after storing current_player from game class

    return jsonify({"status": "AI moved",
                    "current_player": current_player,
                    "winner": winner,
                    "winning_line": winning_line,
                    "draw": draw,
                    "aiRow": row,
                    "aiCol": col})

@app.route('/player-move', methods = ['POST'])
def player_move():
    session_id = session.get('session_id')
    if not session_id:
        return jsonify({"error": "No session!"}), 400

    # assigning an instance of a game object to the game variable
    game = games[session_id]

    # data preparation to game's make_move function, row and col are stored from json frontend
    data = request.get_json()
    row = data["row"]
    col = data["col"]

    try:
        game.make_move(row, col)    # make move is main game class method that involves checking if square(represented as row, col) is taken by any player,
                                    # saving done move to game's board, manages game value properties to represent state of game through values as:
                                    # (game_over, winner, winning_line, current_player)
    except ValueError as e:
        # Return error statement
        return jsonify({"error": str(e)}), 400

    winner = game.winner                # default = None, changing to current_player in case of victory
    winning_line = game.winning_line    # default = None, coordinates of winning line
    draw = False                        # a flag indicating a tie

    current_player = game.current_player    # storing current_player, this is player that just moved
    if game.check_full_board():             # checking is game draw/over or has to be continued
        draw = True
    if game.game_over:
        game.reset_game()
    else:
        game.switch_player()    # method switching current_player from X to O and from O to X, switched player is the next player's move
                                # WARNING: has to be executed after storing current_player from game class

    return jsonify({"status": "Player moved",
                    "current_player": current_player,
                    "winner": winner,
                    "winning_line": winning_line,
                    "draw": draw})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
