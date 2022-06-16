# https://stackoverflow.com/questions/54368686/lstm-having-a-systematic-offset-between-predictions-and-ground-truth
import math

# Paths
DATASET_PATH = "./datasets/"
MODEL_PATH = "./models/"

# Constant variables
PLAYER1 = 1
PLAYER2 = 2

# Hyper-parameters
THINKING_TIME_AGENT1 = 0.5
THINKING_TIME_AGENT2 = 0.5
N_TIME_STEP = 2 # number of past state we keep for our representation

# TicTacToe
#GAME_NAME = "TicTacToe"
#N_ROW = 3
#N_COL = 3
#N_LEVELS = 1
#MAX_GAME_DURATION = math.inf

# Bashni
GAME_NAME = "Bashni"
N_ROW = 8
N_COL = 8
N_LEVELS = 24
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
