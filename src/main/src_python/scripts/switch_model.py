import sys
import os
sys.path.append(os.getcwd()+"/src_python")


from settings.config import MODEL_PATH
from settings.game_settings import GAME_NAME


if __name__ == '__main__':
	oustider_path = MODEL_PATH+GAME_NAME+"_"+"outsider"
	champion_path = MODEL_PATH+GAME_NAME+"_"+"champion"
	old_star = MODEL_PATH+GAME_NAME+"_"+"old_star"

	if os.path.exists(champion_path+".h5") and os.path.exists(oustider_path+".h5"):
		outsider_h5 = oustider_path+".h5"
		champion_h5 = champion_path+".h5"
		old_star_h5 = old_star+".h5"
		
		outsider_onnx = oustider_path+".onnx"
		champion_onnx = champion_path+".onnx"
		old_star_onnx = old_star+".onnx"

		print("--> Replacing champion by old star")
		print("--> Replacing outsider by champion")

		os.rename(champion_h5, old_star_h5)
		os.rename(outsider_h5, champion_h5)	
		
		os.rename(champion_onnx, old_star_onnx)
		os.rename(outsider_onnx, champion_onnx)

		print("--> Done")
