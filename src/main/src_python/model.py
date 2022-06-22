# PROBLEM WHEN RUNNING SCRIPT FROM .sh, ant, or python3...
#from config import *
#from utils import *
from src_python.config import *
from src_python.utils import *


######### Here is the class that contain our AlphaZero model #########

class CustomModel():
	def __init__(self, input_dim, output_dim, n_res_layer, learning_rate, momentum, reg_const):
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.n_res_layer = n_res_layer
		self.learning_rate = learning_rate
		self.momentum = momentum
		self.reg_const = reg_const
		# Define the optimizer
		if OPTIMIZER == "adam":
			opt = tf.keras.optimizers.Adam(
				learning_rate=self.learning_rate,
				beta_1=0.9,
				beta_2=0.999,
				epsilon=1e-07,
				amsgrad=False)
		elif OPTIMIZER == "rmsprop":
			opt = tf.keras.optimizers.RMSprop(
				learning_rate=self.learning_rate,
				rho=0.9,
				momentum=0.0,
				epsilon=1e-07,
				centered=False)
		else:
			opt = tf.keras.optimizers.SGD(
				learning_rate=self.learning_rate, 
				momentum=self.momentum)
		self.opt = opt
		
	def write(self, alphazero_iteration):
		print("--> Saving model for the game :", GAME_NAME, ", AlphaZero iteration :", alphazero_iteration)
		self.model.save(MODEL_PATH+GAME_NAME+str(alphazero_iteration)+".h5")
		
	def summary(self):
		return self.model.summary()
		
	def predict(self, x):
		return self.model.predict(x, verbose=False)

	def fit(self, X, y, n_epochs, batch_size, verbose, validation_split):
		return self.model.fit(
				X, 
				y, 
				epochs=n_epochs, 
				verbose=verbose, 
				validation_split=validation_split, 
				shuffle=True,
				batch_size=batch_size)
			
	def plot_metrics(self, history):
		plt.plot(history.history['policy_head_accuracy'])
		plt.plot(history.history['val_policy_head_accuracy'])
		plt.title('policy head accuracy')
		plt.ylabel('accuracy')
		plt.xlabel('epoch')
		plt.legend(['train', 'val'], loc='upper left')
		plt.show()
		
		plt.plot(history.history['value_head_mean_squared_error'])
		plt.plot(history.history['val_value_head_mean_squared_error'])
		plt.title('value head loss')
		plt.ylabel('mse')
		plt.xlabel('epoch')
		plt.legend(['train', 'val'], loc='upper left')
		plt.show()
		
	# This method builds our entire neural network
	def build_model(self):
		# First layer is in input layer
		input_layer = Input(shape=self.input_dim)
		# Second layer is a classic convolutional layer
		x = self.conv_layer(input_layer, FILTERS, FIRST_KERNEL_SIZE)
		# Then we have several residual layers		
		for _ in range(self.n_res_layer):
			x = self.res_layer(x, FILTERS, KERNEL_SIZE)
		# Then we have two heads, one of policy, one for value
		val_head = self.value_head(x)
		pol_head = self.policy_head(x)
		# Finaly we declare our model
		model = Model(inputs=[input_layer], outputs=[val_head, pol_head])
		model.compile(
			loss={"value_head": "mean_squared_error", "policy_head": softmax_cross_entropy_with_logits},
			loss_weights={"value_head": LOSS_WEIGHTS[0], "policy_head": LOSS_WEIGHTS[1]},
			#metrics={"value_head": "mean_squared_error", "policy_head": "accuracy"},
			optimizer=self.opt)
		self.model = model
		
	# This method returns a classical convolutional layer with batch normalization
	# and a leaky ReLU activation
	def conv_layer(self, x, filters, kernel_size):
		x = Conv2D(
			filters=filters, 
			kernel_size=kernel_size,
			kernel_initializer=KERNEL_INITIALIZER,
			#data_format="channels_first",
			padding="same", 
			use_bias=USE_BIAS, 
			activation=MAIN_ACTIVATION, 
			kernel_regularizer=regularizers.l2(self.reg_const)
		)(x)
		x = BatchNormalization(axis=1)(x)
		x = LeakyReLU()(x)
		return (x)
		
	# This method returns a residual layer with batch normalization
	# and a leaky ReLU activation
	def res_layer(self, input_layer, filters, kernel_size):
		x = self.conv_layer(input_layer, filters, kernel_size)
		x = Conv2D(
			filters=filters, 
			kernel_size=kernel_size, 
			kernel_initializer=KERNEL_INITIALIZER,
			#data_format="channels_first",
			padding="same", 
			use_bias=USE_BIAS, 
			activation=MAIN_ACTIVATION, 
			kernel_regularizer=regularizers.l2(self.reg_const)
		)(x)
		x = BatchNormalization(axis=1)(x)
		x = add([input_layer, x]) # Skip connection
		x = LeakyReLU()(x)
		return (x)
		
	# Our final network will have a common backbone of residual layer and
	# two head, this one is the value head and it will try to predict one
	# scalar value thanks to the input state
	def value_head(self, x):
		x = Conv2D(
			filters=1, # AlphaZero paper
			kernel_size=(1,1), # AlphaZero paper
			kernel_initializer=KERNEL_INITIALIZER,
			#data_format="channels_first",
			padding="same", 
			use_bias=USE_BIAS, 
			activation="linear", 
			kernel_regularizer=regularizers.l2(self.reg_const)
		)(x)
		x = BatchNormalization(axis=1)(x)
		x = LeakyReLU()(x)
		x = Flatten()(x)
		x = Dense(
			NEURONS_VALUE_HEAD, # AlphaZero paper 
			use_bias=USE_BIAS, 
			activation="linear", 
			kernel_regularizer=regularizers.l2(self.reg_const)
		)(x)
		x = LeakyReLU()(x)
		x = Dense(
			1, 
			use_bias=USE_BIAS, 
			activation="tanh", # Value is between -1 and 1 
			kernel_initializer=KERNEL_INITIALIZER,
			kernel_regularizer=regularizers.l2(self.reg_const),
			name="value_head"
		)(x)
		return (x)
		
	# This one is our policy head and will predict a distribution over moves
	# which will be our new policy
	def policy_head(self, x):
		x = Conv2D(
			filters=2, # AlphaZero paper
			kernel_size=(1,1), # AlphaZero paper 
			kernel_initializer=KERNEL_INITIALIZER,
			#data_format="channels_first",
			padding="same", 
			use_bias=USE_BIAS, 
			activation="linear", 
			kernel_regularizer=regularizers.l2(self.reg_const)
		)(x)
		x = BatchNormalization(axis=1)(x)
		x = LeakyReLU()(x)
		x = Flatten()(x)
		x = Dense(
			self.output_dim, 
			use_bias=USE_BIAS, 
			activation="linear", # We are a softmax cross entropy with logits loss
			kernel_initializer=KERNEL_INITIALIZER,
			kernel_regularizer=regularizers.l2(self.reg_const),
			name="policy_head"
		)(x)
		return (x)
		
