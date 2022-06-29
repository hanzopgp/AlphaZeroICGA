import sys
import os
sys.path.append(os.getcwd()+"/src_python")


from utils import *


outsider = MODEL_PATH+GAME_NAME+"_"+"outsider"+".h5"
champion = MODEL_PATH+GAME_NAME+"_"+"champion"+".h5"
old_star = MODEL_PATH+GAME_NAME+"_"+"old_star"+".h5"

print("--> Replacing champion by old star")
print("--> Replacing outsider by champion")

os.rename(champion, old_star)
os.rename(outsider, champion)	

print("--> Done")
