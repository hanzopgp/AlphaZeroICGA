import tensorflow as tf

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

NUM_DOJO = 2
MAX_ITERATION_AGENTS_DOJO = 30 # 300
THINKING_TIME_AGENTS_DOJO = -1

NUM_EPISODE = 3
MAX_ITERATION_AGENT1 = 30
MAX_ITERATION_AGENT2 = 30
THINKING_TIME_AGENT1 = -1
THINKING_TIME_AGENT2 = -1

N_EPOCHS = 15

MAX_GAME_DURATION = -1

######### CONSTANT VARIABLES #########

DEBUG_PRINT = True
PROFILING_ACTIVATED = True
PLAYER1 = 1
PLAYER2 = 2
DATASET_PATH = "./datasets/"
MODEL_PATH = "./models/"
WINNERS_FILE="./models/save_winners.txt"
N_MOVES_TYPICAL_POSITION_BASHNI = 20 # need to think a bit more about this one !	
OUTSIDER_MIN_WINRATE = 0.55

######### MCTS PARAMETERS #########

CSTE_PUCT = 1 # no idea about the value of this variable in the paper 
DIRICHLET_ALPHA = 10/N_MOVES_TYPICAL_POSITION_BASHNI # noise in the estimated policy -> more exploration
WEIGHTED_SUM_DIR = 0.75 # this value comes from the paper
TEMPERATURE = 1 # 1 -> no change, 0 -> argmax
MAX_SAMPLE = 10000 # can decide the max size of the dataset 

######### NN parameters #########

TRAIN_SAMPLE_SIZE = 2048
RANDOM_SEED = 42
BATCH_SIZE = 256
VERBOSE = 1
VALIDATION_SPLIT = 0.2
LOSS_WEIGHTS = [0.5, 0.5] # first one is value, second one policy

MAIN_ACTIVATION = "relu"
FILTERS = 64
KERNEL_SIZE = (3,3)
KERNEL_INITIALIZER = tf.keras.initializers.GlorotNormal() # Xavier uniform
FIRST_KERNEL_SIZE = (3,3)
USE_BIAS = False
N_RES_LAYER = 10
NEURONS_VALUE_HEAD = 32 # number of neurons in last dense layer

OPTIMIZER = "adam"
LEARNING_RATE = 0.1
MOMENTUM = 0.9
REG_CONST = 1e-4 # L2 reg

######### OPTIMIZATION VARIABLES #########

INDEX_ACTION_TAB_SIGN = [[1,1], [-1,1], [1,-1], [-1,-1]]




