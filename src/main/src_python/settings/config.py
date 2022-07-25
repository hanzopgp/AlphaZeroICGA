######### GAME DEPENDANT VARIABLES #########

MAX_GAME_DURATION = -1 # Max episode or dojo duration in seconds

# MAX_GAME_MOVES = 200 # Bashni
# MAX_GAME_MOVES = 400 # Plot
# MAX_GAME_MOVES = 200 # Quoridor
# MAX_GAME_MOVES = 400 # Mini Wars
# MAX_GAME_MOVES = 200 # Plakoto
# MAX_GAME_MOVES = 200 # Lotus
MAX_GAME_MOVES = 50 # Connect Four

######### TIME CONSUMING VARIABLES #########

ONNX_INFERENCE = True # ONNX inference should be False if using GPU

N_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10

NUM_EPISODE = 10 # Number of self play games by worker
VANILLA_EPISODE_MULTIPLIER = 5 # Factor by which we multiply the number of episodes when vanilla MCTS are playing
MAX_ITERATION_AGENT = 100 # Max number of nodes discovered by the MCTS
THINKING_TIME_AGENT = -1 # Max number of seconds for the MCTS to run

NUM_DOJO = 4
MAX_ITERATION_AGENTS_DOJO = 100 
THINKING_TIME_AGENTS_DOJO = -1

N_BATCH_PREDICTION = 5 # Number of batch per MCTS simulation
MINIMUM_QUEUE_PREDICTION = MAX_ITERATION_AGENT//N_BATCH_PREDICTION + 1 # Max number of nodes discovered before computing the values estimation with our model

######### CONSTANT VARIABLES #########

DEBUG_PRINT = True # Prints additional informations during the self play and scripts
PROFILING_ACTIVATED = False # Prints additional informations such as the time per function
PLAYER1 = 1
PLAYER2 = 2
N_PLAYERS = 2 + 1
DATASET_PATH = "./datasets/"
MODEL_PATH = "./models/"
WINNERS_FILE="./models/save_winners.txt"
OUTSIDER_MIN_WINRATE = 0.6
MAX_SIZE_FULL_DATASET = 50_000 # Maximum number of examples in the dataset
RATIO_TRAIN = 1/2 # If 2/3 --> always train on the last third of the dataset

# N_MOVES_TYPICAL_POSITION_BASHNI = 15
N_MOVES_TYPICAL_POSITION_CONNECTFOUR = 6

######### MCTS PARAMETERS #########

CSTE_PUCT = 2 # Exploration constant 
MAX_SAMPLE = 10_000 # Can decide the max size of the dataset per iteration 
WEIGHTED_SUM_DIR = 0.75 # this value comes from the paper
DIRICHLET_ALPHA = 10/N_MOVES_TYPICAL_POSITION_CONNECTFOUR # noise in the estimated policy -> more exploration
TEMPERATURE = 1 # 1 -> no change, 0 -> argmax 

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
NEURONS_VALUE_HEAD = 128 # Number of neurons in last dense layer

OPTIMIZER = "sgd"
LEARNING_RATE_DECAY_IT = 5 # LR decay every 5 alphazero iteration
LEARNING_RATE_DECAY_FACTOR = 2 # Divided by 2 each time
BASE_LEARNING_RATE = 0.1
MOMENTUM = 0.9
REG_CONST = 1e-5 # L2 reg

LOSS_WEIGHTS = [0.5, 0.5]

######### OPTIMIZATION VARIABLES #########

INDEX_ACTION_TAB_SIGN = [[1,1], [-1,1], [1,-1], [-1,-1]]




