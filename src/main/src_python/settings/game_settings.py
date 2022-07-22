######### BASHNI PARAMETERS #########

# # General
# GAME_NAME = "Bashni"
# # State representation
# N_ROW = 8
# N_COL = 8
# N_LEVELS = 24
# N_ADDITIONAL_FEATURES = 1 # currently only the color of the current player
# N_TIME_STEP = 3 # number of past state we keep for our representation
# N_REPRESENTATION_STACK = N_ADDITIONAL_FEATURES + (N_TIME_STEP * 2) * N_LEVELS 
# # Action representation
# N_DISTANCE = 8 # the king can move on the whole diagonal
# N_ORIENTATION = 4 # can go only left or right diagonaly, for both players
# N_ACTION_STACK = N_ORIENTATION * N_DISTANCE

######### PLOY PARAMETERS #########

# # General
# GAME_NAME = "Ploy"
# # State representation
# N_ROW = 9
# N_COL = 9
# N_LEVELS = 8 # here levels are the type of pawn (shields, probes, lances, commander) (there are different type inside those types)
# N_ADDITIONAL_FEATURES = 1
# N_TIME_STEP = 3
# N_REPRESENTATION_STACK = N_ADDITIONAL_FEATURES + (N_TIME_STEP * 2) * N_LEVELS 
# # Action representation
# N_DISTANCE = 3 # max steps (for the lance) 
# N_ORIENTATION = 4 # max freedom (for the commander)
# N_ROTATION = 2 # pieces can rotate left or right
# N_ACTION_STACK = N_ORIENTATION * N_DISTANCE * N_DISTANCE

######### QUORIDOR PARAMETERS #########

# # General
# GAME_NAME = "Quoridor"
# # State representation
# N_ROW = 13
# N_COL = 13
# N_LEVELS = 2 # here levels are the type of place (pawn or wall)
# N_ADDITIONAL_FEATURES = 1
# N_TIME_STEP = 3
# N_REPRESENTATION_STACK = N_ADDITIONAL_FEATURES + (N_TIME_STEP * 2) * N_LEVELS 
# # Action representation
# N_DISTANCE = 1 # max steps (for the pawn) 
# N_ORIENTATION = 4 # max freedom (for the pawn)
# N_ROTATION = 2 # can move the pawn or place a wall
# N_ACTION_STACK = N_ORIENTATION * N_DISTANCE * N_DISTANCE

######### MINIWARS PARAMETERS #########

# # General
# GAME_NAME = "MiniWars"
# # State representation
# N_ROW = 8
# N_COL = 8
# N_LEVELS = 18 # here levels are the type of pieces
# N_ADDITIONAL_FEATURES = 1
# N_TIME_STEP = 3
# N_REPRESENTATION_STACK = N_ADDITIONAL_FEATURES + (N_TIME_STEP * 2) * N_LEVELS 
# # Action representation
# N_DISTANCE = 1 # max steps (for the pawn) 
# N_ORIENTATION = 4 # max freedom (for the pawn)
# N_ROTATION = 2 # can move the pawn or place a wall
# N_ACTION_STACK = N_ORIENTATION * N_DISTANCE * N_DISTANCE

######### PLAKOTO PARAMETERS #########

# # General
# GAME_NAME = "Plakoto"
# # State representation
# N_ROW = 6
# N_COL = 6
# N_LEVELS = 15 + 1 # levels + one level for dice
# N_ADDITIONAL_FEATURES = 1
# N_TIME_STEP = 3
# N_REPRESENTATION_STACK = N_ADDITIONAL_FEATURES + (N_TIME_STEP * 2) * N_LEVELS 
# # Action representation
# N_DISTANCE = 12 # double dice roll 
# N_ORIENTATION = 2 # go forward or go in enemy lines
# N_ROTATION = 1 
# N_ACTION_STACK = N_ORIENTATION * N_DISTANCE * N_DISTANCE

######### LOTUS PARAMETERS #########

# # General
# GAME_NAME = "Lotus"
# # State representation
# N_ROW = 9
# N_COL = 9
# N_LEVELS = 1
# N_ADDITIONAL_FEATURES = 1
# N_TIME_STEP = 3
# N_REPRESENTATION_STACK = N_ADDITIONAL_FEATURES + (N_TIME_STEP * 2) * N_LEVELS 
# # Action representation
# N_DISTANCE = 12 
# N_ORIENTATION = 2 
# N_ROTATION = 1 
# N_ACTION_STACK = N_ORIENTATION * N_DISTANCE * N_DISTANCE

######### CONNECT FOUR PARAMETERS #########

# General
GAME_NAME = "ConnectFour"
# State representation
N_ROW = 6
N_COL = 7
N_LEVELS = 1
N_ADDITIONAL_FEATURES = 1
N_TIME_STEP = 3
N_REPRESENTATION_STACK = N_ADDITIONAL_FEATURES + (N_TIME_STEP * 2) * N_LEVELS 
# Action representation
N_DISTANCE = -1 
N_ORIENTATION = -1 
N_ROTATION = -1 
N_ACTION_STACK = N_ROW