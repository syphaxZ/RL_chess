import numpy as np
import chess
import chess.svg
import chess.engine
from stable_baselines3 import DQN

# Charger le modèle pré-entraîné
model = DQN.load("dqn_chess_model")

def play_game():
    # Initialiser le jeu d'échecs
    board = chess.Board()
    # Variable pour suivre le joueur actuel
    player_turn = 0

    while not board.is_game_over():
        print(board)

        if player_turn == 0:
            # Tour du joueur humain
            human_move = input("Entrez votre coup (notation algébrique): ")
            # Vérifier si le coup est valide
            if chess.Move.from_uci(human_move) in board.legal_moves:
                board.push_uci(human_move)
                player_turn = 1
            else:
                print("Coup invalide. Veuillez réessayer.")
                continue
        else:
            # Tour du modèle
            action, _ = model.predict(board_to_observation(board))
            predicted_move = chess.Move.from_uci(action)
            if predicted_move in board.legal_moves:
                board.push(predicted_move)
                player_turn = 0
            else:
                print("Le modèle a choisi un coup invalide. Fin de la partie.")
                break

    print("Partie terminée.")
    print("Résultat de la partie:", board.result())

def board_to_observation(board):
    # Convertir l'état du plateau en une observation utilisable par le modèle
    observation = np.zeros(64, dtype=np.float32)
    for i in range(64):
        piece = board.piece_at(i)
        if piece is not None:
            observation[i] = piece.piece_type + (6 * piece.color)
    return observation

# Lancer le jeu
play_game()
