######### ALPHAZERO PAPER VARIABLES #########

# thinking_time 40ms
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

######### TIME CONSUMING VARIABLES #########

ONNX_INFERENCE = True

N_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 20

NUM_DOJO = 2
MAX_ITERATION_AGENTS_DOJO = 50 
THINKING_TIME_AGENTS_DOJO = -1

NUM_EPISODE = 2
MAX_ITERATION_AGENT = 50
THINKING_TIME_AGENT = -1

MAX_GAME_DURATION = -1
MAX_GAME_MOVES = 200

######### CONSTANT VARIABLES #########

DEBUG_PRINT = True
PROFILING_ACTIVATED = False
PLAYER1 = 1
PLAYER2 = 2
DATASET_PATH = "./datasets/"
MODEL_PATH = "./models/"
WINNERS_FILE="./models/save_winners.txt"
N_MOVES_TYPICAL_POSITION_BASHNI = 15
OUTSIDER_MIN_WINRATE = 0.55

######### MCTS PARAMETERS #########

CSTE_PUCT = 2 # exploration constant 
MAX_SAMPLE = 10000 # can decide the max size of the dataset 

######### NN parameters #########

TRAIN_SAMPLE_SIZE = 4096
RANDOM_SEED = 42
BATCH_SIZE = 512
VERBOSE = 1
VALIDATION_SPLIT = 0.25

MAIN_ACTIVATION = "relu"
FILTERS = 64
KERNEL_SIZE = (3,3)
FIRST_KERNEL_SIZE = (3,3)
USE_BIAS = True
N_RES_LAYER = 5
NEURONS_VALUE_HEAD = 128 # number of neurons in last dense layer

OPTIMIZER = "sgd"
LEARNING_RATE_DECAY_IT = 5 # LR decay every 5 alphazero iteration
LEARNING_RATE_DECAY_FACTOR = 5 # divided by 5 each time
BASE_LEARNING_RATE = 0.1
MOMENTUM = 0.9
REG_CONST = 1e-5 # L2 reg

######### OPTIMIZATION VARIABLES #########

INDEX_ACTION_TAB_SIGN = [[1,1], [-1,1], [1,-1], [-1,-1]]




