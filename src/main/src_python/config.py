import math

# AlphaZero paper config for chess
# thiking_time 40ms
# number of trials 800
# total number of games 44M
# learning rate 0.2, 0.02, 0.002, 0.0002

# Paths
DATASET_PATH = "./datasets/"
MODEL_PATH = "./models/"

# Constant variables
PLAYER1 = 1
PLAYER2 = 2

# Hyper-parameters
THINKING_TIME_AGENT1 = 0.5
THINKING_TIME_AGENT2 = 0.5
MAX_ITERATION_AGENT1 = -1
MAX_ITERATION_AGENT2 = -1
N_TIME_STEP = 2 # number of past state we keep for our representation

# Bashni
GAME_NAME = "Bashni"
N_ROW = 8
N_COL = 8
N_LEVELS = 24
N_DISTANCE = 7 # the king can move on the whole diagonal
N_ORIENTATION = 4 # can go only left or right diagonaly, for both players
N_ACTION_STACK = N_ORIENTATION * N_DISTANCE
N_ADDITIONAL_FEATURES = 1 # currently only the color of the current player
N_REPRESENTATION_STACK = N_ADDITIONAL_FEATURES + N_TIME_STEP * 2 * N_LEVELS 
MAX_MOVES_POSSIBLE = N_ROW*N_COL*N_ACTION_STACK*12 # 12 pieces
MAX_GAME_DURATION = math.inf

# Trial parameters
NUM_TRIALS = 1
MAX_SAMPLE = NUM_TRIALS * N_ROW * N_COL * N_LEVELS + 1	

# NN parameters
RANDOM_SEED = 42
N_EPOCHS = 20
BATCH_SIZE = 128
VERBOSE = True
VALIDATION_SPLIT = 0.2
LOSS_WEIGHTS = [0.33, 0.67] # first one is value, second one policy

FILTERS = 128
KERNEL_SIZE = (3,3)
FIRST_KERNEL_SIZE = (5,5) 
N_RES_LAYER = 10
NEURONS_VALUE_HEAD = 128 # number of neurons in last dense layer

OPTIMIZER = "adam"
LEARNING_RATE = 1e-2
MOMENTUM = 0.9
REG_CONST = 1e-4 # L2 reg
