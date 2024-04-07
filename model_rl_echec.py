import chess
import chess.svg
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
import gymnasium as gym  # Modification ici
from gymnasium import spaces  # Modification ici

class ChessEnv(gym.Env):  # Notez que c'est maintenant gymnasium.Env
    """Un environnement personnalisé qui suit l'interface gymnasium."""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(ChessEnv, self).__init__()
        self.action_space = spaces.Discrete(64 * 64)  # Exemple simplifié
        # Définir l'espace d'observation ici
        self.observation_space = spaces.Box(low=0, high=12, shape=(64,), dtype=np.float32)
        # Initialiser le plateau d'échecs
        self.board = chess.Board()

    def step(self, action):
        try:
            move = self._action_to_move(action)
            self.board.push(move)  # Effectue le mouvement
            done = self.board.is_game_over()
            reward = 1.0 if self.board.is_checkmate() else 0.0  # Récompense simplifiée
            info = {}

            # Vérifiez si la partie est terminée pour distinguer entre 'terminated' et 'truncated'.
            # Dans le contexte des échecs, nous pouvons considérer toutes les fins de jeu comme 'terminated'.
            # 'truncated' pourrait être utilisé si vous implémentez une limite de coups.
            terminated = done
            truncated = False  # Modifiez cette logique selon vos besoins.

            return self._board_to_observation(), reward, terminated, truncated, info
        except Exception as e:
            # En cas d'erreur (par exemple, mouvement illégal), vous pourriez vouloir ajuster la récompense ou les flags.
            return self._board_to_observation(), 0.0, True, False, {'error': str(e)}

    def reset(self, seed=None, return_info=False, options=None):
    # Si vous voulez utiliser le seed, vous pouvez l'ajouter ici pour initialiser votre générateur de nombres aléatoires.
    # Par exemple:
    # self.np_random, seed = seeding.np_random(seed)
    
        self.board.reset()
        observation = self._board_to_observation()
    
        if return_info:
            info = {}  # Vous pouvez inclure des informations supplémentaires ici si nécessaire.
            return observation, info
        return observation, {}



    def render(self, mode='human', close=False):
        if mode == 'human':
            print(self.board)

    def _action_to_move(self, action):
        # Convertit une action en un mouvement d'échecs
        from_square = chess.SQUARES[int(action / 64)]
        to_square = chess.SQUARES[action % 64]
        return chess.Move(from_square, to_square)

    def _board_to_observation(self):
        # Convertit l'état du plateau en un tableau 1D
        observation = np.zeros(64, dtype=np.float32)
        for i in range(64):
            piece = self.board.piece_at(i)
            if piece is not None:
                observation[i] = piece.piece_type + (6 * piece.color)
        return observation

# Vérification de l'environnement
env = ChessEnv()
check_env(env)


from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

# Initialisation de l'environnement vectorisé
vec_env = make_vec_env(lambda: ChessEnv(), n_envs=1)

# Création du modèle DQN
model = DQN("MlpPolicy", vec_env, verbose=1, learning_rate=1e-3, buffer_size=10000, learning_starts=1000, batch_size=32, tau=1.0, gamma=0.99, train_freq=4, gradient_steps=1, optimize_memory_usage=False, policy_kwargs=dict(net_arch=[256, 256]), tensorboard_log="./chess_dqn/")

# Configuration du callback pour sauvegarder le modèle
checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./chess_dqn/', name_prefix='dqn_chess_model')

# Entraînement du modèle
model.learn(total_timesteps=50000, callback=checkpoint_callback)

# Sauvegarde du modèle
model.save("dqn_chess_model")

# Chargement du modèle
# model = DQN.load("dqn_chess_model", env=vec_env)
