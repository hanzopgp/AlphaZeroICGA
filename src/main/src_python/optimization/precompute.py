import os
import sys
print(os.getcwd())
sys.path.append(os.getcwd()+"/src_python")


from utils import *


def precompute_action_index():
	pre_action_index = np.zeros((N_ROW*N_COL, N_ROW*N_COL))
	for from_ in range(N_ROW*N_COL):
		for to_ in range(N_ROW*N_COL):
			pre_action_index[from_][to_] = index_action(from_, to_)
	return pre_action_index
	
def precompute_reverse_action_index():
	n_returns = 2
	pre_reverse_action_index = np.zeros((N_ROW, N_COL, N_ACTION_STACK, n_returns))
	for to_x in range(N_ROW):
		for to_y in range(N_COL):
			for action_index in range(N_ACTION_STACK):
				for index_return in range(n_returns):
					pre_reverse_action_index[to_x][to_y][action_index][index_return] = reverse_index_action(to_x, to_y, action_index)[index_return]
	return pre_reverse_action_index
	
def precompute_get_coord():
	n_returns = 4
	pre_coords = np.zeros((N_ROW*N_COL, N_ROW*N_COL, n_returns))
	for from_ in range(N_ROW*N_COL):
		for to_ in range(N_ROW*N_COL):
			for index_return in range(n_returns):
				pre_coords[from_][to_][index_return] = get_coord(from_, to_)[index_return]
	return pre_coords
	
def precompute_all():
	return precompute_action_index(), precompute_reverse_action_index(), precompute_get_coord()
	
	
#print(precompute_action_index().shape)
#print(precompute_reverse_action_index().shape)
#print(precompute_get_coord().shape)

#print(precompute_reverse_action_index()[1][0][0][0])
#print(precompute_reverse_action_index()[1][0][0][1])

#print(precompute_get_coord()[1][12][0])
#print(precompute_get_coord()[1][12][1])
#print(precompute_get_coord()[1][12][2])
#print(precompute_get_coord()[1][12][3])
