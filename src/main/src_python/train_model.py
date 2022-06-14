from config import *
from model import CustomModel

		
print("ok bg")
model = CustomModel((6,3,3), (9), 3, 1e-3, 1e-4, 1e-6)
model.build_model()
#model.summary()
