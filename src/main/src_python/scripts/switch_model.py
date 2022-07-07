import sys
import os
sys.path.append(os.getcwd()+"/src_python")


from settings.config import MODEL_PATH
from settings.game_settings import GAME_NAME


if __name__ == '__main__':
	outsider_h5 = MODEL_PATH+GAME_NAME+"_"+"outsider"+".h5"
	champion_h5 = MODEL_PATH+GAME_NAME+"_"+"champion"+".h5"
	old_star_h5 = MODEL_PATH+GAME_NAME+"_"+"old_star"+".h5"
	
	outsider_onnx = MODEL_PATH+GAME_NAME+"_"+"outsider"+".onnx"
	champion_onnx = MODEL_PATH+GAME_NAME+"_"+"champion"+".onnx"
	old_star_onnx = MODEL_PATH+GAME_NAME+"_"+"old_star"+".onnx"

	print("--> Replacing champion by old star")
	print("--> Replacing outsider by champion")

	os.rename(champion_h5, old_star_h5)
	os.rename(outsider_h5, champion_h5)	
	
	os.rename(champion_onnx, old_star_onnx)
	os.rename(outsider_onnx, champion_onnx)

	print("--> Done")
