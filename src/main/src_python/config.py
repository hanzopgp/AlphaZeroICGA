import math

# AlphaZero paper config for chess
# thiking_time 40ms
# number of trials 800
# total number of games 44M
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
MAX_MOVES_POSSIBLE = N_ROW * N_COL * N_ACTION_STACK * 12 # 12 pieces

######### MCTS PARAMETERS #########
# games, row, col, features
# (500, 8, 8, 192) --> too big dataset  
# (300, 8, 8, 144) --> better

THINKING_TIME_AGENT1 = 1
THINKING_TIME_AGENT2 = 1
MAX_ITERATION_AGENT1 = -1
MAX_ITERATION_AGENT2 = -1
NUM_TRIALS = 300 # 800 games ~ 80 000 moves ~ 10 hours
MAX_GAME_DURATION = 250 # 200 seconds is fine
MAX_SAMPLE = 100000 # can decide the size of the dataset 

######### NN parameters #########

RANDOM_SEED = 42
N_EPOCHS = 10
BATCH_SIZE = 256
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
