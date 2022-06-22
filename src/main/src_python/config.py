import math

######### ALPHAZERO PAPER VARIABLES #########

# AlphaZero paper config for chess
# thiking_time 40ms
# total number of games 44M
# number of trials 25 000
# sample 2048 positions from the last 500 000 games
# evaluate network after 1 000 training loops
# play 400 games to evaluate new network
# need more than 55% winrate to pass
# 7 time step for each player

# learning rate 0.2, 0.02, 0.002, 0.0002
# kernel size (3,3)
# n residual layers 40
# n filters 256
# mlp neurons 256

######### CONSTANTE VARIABLES #########

PLAYER1 = 1
PLAYER2 = 2
DATASET_PATH = "./datasets/"
MODEL_PATH = "./models/"

######### DOJO PARAMETERS #########

NUM_DOJO = 1
THINKING_TIME_AGENTS_DOJO = 0.1
OUTSIDER_MIN_WINRATE = 0.5 

######### GAME PARAMETERS #########

GAME_NAME = "Bashni"
N_ROW = 8
N_COL = 8
N_LEVELS = 24
N_TIME_STEP = 3 # number of past state we keep for our representation
N_DISTANCE = 8 # the king can move on the whole diagonal
N_ORIENTATION = 4 # can go only left or right diagonaly, for both players
N_ACTION_STACK = N_ORIENTATION * N_DISTANCE
N_ADDITIONAL_FEATURES = 1 # currently only the color of the current player
N_REPRESENTATION_STACK = N_ADDITIONAL_FEATURES + (N_TIME_STEP * 2) * N_LEVELS 
#MAX_MOVES_POSSIBLE = N_ROW * N_COL * N_ACTION_STACK * 12 # 12 pieces
N_LEGAL_MOVES = N_ACTION_STACK * 12 # 12 pieces

######### MCTS PARAMETERS #########
# games, row, col, features
# (500, 8, 8, 193) --> too big dataset  
# (200, 8, 8, 145) --> better
DIRICHLET_ALPHA = 10/N_LEGAL_MOVES # noise in the estimated policy -> more exploration
WEIGHTED_SUM_DIR = 0.75 # this value comes from the paper
TEMPERATURE = 1 # 1 -> no change, 0 -> argmax
THINKING_TIME_AGENT1 = 0.1
THINKING_TIME_AGENT2 = 0.1
MAX_ITERATION_AGENT1 = -1
MAX_ITERATION_AGENT2 = -1
NUM_TRIALS = 1 # 200 games ~ 26000 moves ~ 446 minutes (1s TT)
MAX_GAME_DURATION = 300 # 300 seconds is fine
MAX_SAMPLE = 100000 # can decide the size of the dataset 

######### NN parameters #########

RANDOM_SEED = 42
N_EPOCHS = 1
BATCH_SIZE = 512 # 256 working
VERBOSE = True
VALIDATION_SPLIT = 0.2
LOSS_WEIGHTS = [0.5, 0.5] # first one is value, second one policy

MAIN_ACTIVATION = "relu"
FILTERS = 32
KERNEL_SIZE = (3,3)
KERNEL_INITIALIZER = "random_normal"
FIRST_KERNEL_SIZE = (3,3) 
USE_BIAS = False
N_RES_LAYER = 5
NEURONS_VALUE_HEAD = 64 # number of neurons in last dense layer

OPTIMIZER = "adam"
LEARNING_RATE = 0.002
MOMENTUM = 0.9
REG_CONST = 1e-4 # L2 reg
