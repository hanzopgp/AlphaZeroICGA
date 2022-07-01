import sys
import os
sys.path.append(os.getcwd()+"/src_python")
import re


from config import *
from utils import *


def read_txts():
	outsider_winrate = 0
	total = 0

	f = []
	for (dirpath, dirnames, filenames) in os.walk(MODEL_PATH):
	    f.extend(filenames)

	for path in f:
		with open(MODEL_PATH+path, "r") as file:
			if "txt" in MODEL_PATH+path and "save" not in MODEL_PATH+path:
				first_line = file.readline()
				outsider_winrate += float(re.findall("\d+\.\d+", first_line)[0])
				#print(MODEL_PATH+path," ",outsider_winrate)
				total += 1
				Popen("rm "+MODEL_PATH+path, shell=True).wait()
	final_winrate = outsider_winrate/total
	#print(final_winrate)
	write_winner(final_winrate)

if __name__ == '__main__':
	read_txts()
