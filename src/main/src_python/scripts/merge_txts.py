import sys
import os
import re
from subprocess import Popen
sys.path.append(os.getcwd()+"/src_python")


from settings.config import MODEL_PATH
from utils import write_winner


def read_txts():
	outsider_winrate = 0
	total = 0

	f = []
	for (dirpath, dirnames, filenames) in os.walk(MODEL_PATH):
	    f.extend(filenames)

	for path in f:
		with open(MODEL_PATH+path, "r") as file:
			if "txt" in MODEL_PATH+path and "save" not in MODEL_PATH+path:
				print("--> Reading file :", MODEL_PATH+path)
				first_line = file.readline()
				outsider_winrate += float(re.findall("\d+\.\d+", first_line)[0])
				total += 1
				Popen("rm "+MODEL_PATH+path, shell=True).wait()
	final_winrate = outsider_winrate/total
	write_winner(final_winrate)


if __name__ == '__main__':
	read_txts()
