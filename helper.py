from tensorflow import keras
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def build_model(inputShape, width, height):
	# further improvements:
	# - take width, height(build a loop)
	# - take optimizer, loss || assign default parameters

	model = keras.Sequential()
	model.add(keras.layers.Dense(width, input_shape=(inputShape,), activation=tf.nn.relu))

	for i in range(height):
		model.add(keras.layers.Dense(width, activation=tf.nn.relu))

	model.add(keras.layers.Dense(1))

	model.compile(optimizer='Adagrad', loss='msle', metrics=['msle'])

	return model

class PrintProgress(keras.callbacks.Callback):
	#print for each epoch to keep track of training progress
	def on_epoch_end(self,epoch,logs):
		print("{}th epoch, loss={}".format(epoch, logs.get('loss')), end="\r")


def plot_history(history):
	#plotting error
	plt.figure()
	plt.xlabel('Epoch')
	plt.ylabel('Mean Squared Logarithmic Error')
	plt.plot(history.epoch, np.array(history.history['mean_squared_logarithmic_error']), 
			label='Train Loss')
	plt.plot(history.epoch, np.array(history.history['val_mean_squared_logarithmic_error']),
			label = 'Val loss')
	plt.legend()
	plt.show() 

def output_file_creator(name_out, ID, predictions1, predictions2):
	myOutput = ID.join(predictions1)
	myOutput = myOutput.join(predictions2)
	myOutput.to_csv(name_out + '.csv', index=False)
	return myOutput


def build_model2_layered(inputShape, layer1, layer2):
	# further improvements:
	# - take width, height(build a loop)
	# - take optimizer, loss || assign default parameters

	model = keras.Sequential()
	model.add(keras.layers.Dense(layer1, input_shape=(inputShape,), activation=tf.nn.relu))
	model.add(keras.layers.Dense(layer2, activation=tf.nn.relu))
	model.add(keras.layers.Dense(1))

	model.compile(optimizer='Adagrad', loss='msle', metrics=['msle'])

	return model

def build_model3_layered(inputShape, layer1, layer2, layer3):
	# further improvements:
	# - take width, height(build a loop)
	# - take optimizer, loss || assign default parameters

	model = keras.Sequential()
	model.add(keras.layers.Dense(layer1, input_shape=(inputShape,), activation=tf.nn.relu))
	model.add(keras.layers.Dense(layer2, activation=tf.nn.relu))
	model.add(keras.layers.Dense(layer3, activation=tf.nn.relu))

	model.add(keras.layers.Dense(1))

	model.compile(optimizer='Adagrad', loss='msle', metrics=['msle'])

	return model