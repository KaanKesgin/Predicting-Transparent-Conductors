import tensorflow as tf
import pandas as pd 
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt

from tensorflow import keras
from math import sqrt

from helper import build_model
from helper import PrintProgress
from helper import plot_history
from helper import output_file_creator

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings("ignore")

#Load the Data
trainSet = pd.read_csv('data/train.csv')
trainSet.drop('id', axis=1, inplace=True); trainSet.drop('spacegroup', axis=1, inplace=True);

testSet = pd.read_csv('data/test.csv')
ID = pd.DataFrame(testSet.id)
testSet.drop('id', axis=1, inplace=True); testSet.drop('spacegroup', axis=1, inplace=True);

#Scale the Data
scalerFeatures, scalerFormation, scalerBandgap = MinMaxScaler(), MinMaxScaler(), MinMaxScaler()
features, labelsFormation, labelsBandgap = trainSet.iloc[:,:-2], trainSet.iloc[:,-2], trainSet.iloc[:,-1]

features = pd.DataFrame(scalerFeatures.fit_transform(features))
labelsFormation = pd.DataFrame(scalerFormation.fit_transform(labelsFormation))
labelsBandgap = pd.DataFrame(scalerBandgap.fit_transform(labelsBandgap))

#Split the Data
X_trainF, X_testF, y_trainF, y_testF = train_test_split(features, labelsFormation, test_size=0.20, random_state=42)
X_trainB, X_testB, y_trainB, y_testB = train_test_split(features, labelsBandgap, test_size=0.20, random_state=42)

#Build the model
input_shape = features.shape[1]
n_epochs = 5000
width = 30
height = 10

bestRMSLEForm = [0.051880246254638995, 28, 0]
bestRMSLEBand = [0.03375373930387272, 25, 0]

for j in range(1,height+1):

	for i in range(1,width+1):

		modelForm = build_model(input_shape, i, j)
		modelBand = build_model(input_shape, i, j)

	#Train the model
		print("training for width {}, height {} has started".format(i,j))

		early_stopForm = keras.callbacks.EarlyStopping(monitor='val_loss', patience=200)
		historyForm = modelForm.fit(X_trainF, y_trainF, epochs=n_epochs, validation_split=0.2, verbose=0, 
			callbacks=[early_stopForm, PrintProgress()])

		early_stopBand = keras.callbacks.EarlyStopping(monitor='val_loss', patience=200)
		historyBand = modelBand.fit(X_trainB, y_trainB, epochs=n_epochs, validation_split=0.2, verbose=0, 
			callbacks=[early_stopBand, PrintProgress()])

	#Plot the model
		#plot_history(historyForm)
		#plot_history(historyBand)

	#See how well the model did on test data
		[lossF, msleF] = modelForm.evaluate(X_testF, y_testF, verbose=0)
		[lossB, msleB] = modelBand.evaluate(X_testB, y_testB, verbose=0)
		log_rmsleForm, log_rmsleBand = sqrt(msleF), sqrt(msleB)

		if log_rmsleForm<bestRMSLEForm[0]:
			bestRMSLEForm[0] = log_rmsleForm
			bestRMSLEForm[1] = i
			bestRMSLEForm[2] = j
			modelForm.save_weights('best_weights_formation.h5')

		if log_rmsleBand<bestRMSLEBand[0]:
			bestRMSLEBand[0] = log_rmsleBand
			bestRMSLEBand[1] = i
			bestRMSLEBand[2] = j
			modelBand.save_weights('best_weights_bandgap.h5')


		print("Best RMSLE for formation energy so far is for depth={}, height={}, is: {}".format(bestRMSLEForm[1], bestRMSLEForm[2],bestRMSLEForm[0]))
		print("Best RMSLE for bandgap energy so far is for depth={}, height={}, is: {}".format(bestRMSLEBand[1], bestRMSLEBand[2], bestRMSLEBand[0]))
		print("Best combined RMSLE so far is: {}".format(bestRMSLEForm[0]+bestRMSLEBand[0]))


#Predict with test set
modelForm = build_model(input_shape, 26, 1)
modelBand = build_model(input_shape, 26, 1)

modelForm.load_weights('best_weights_formation.h5')
modelBand.load_weights('best_weights_bandgap.h5')

testSet = pd.DataFrame(scalerFeatures.fit_transform(testSet))
predictionForm = modelForm.predict(testSet).flatten()
predictionForm = pd.DataFrame(scalerFormation.inverse_transform(predictionForm), columns = ['formation_energy_ev_natom'])

predictionBand = modelBand.predict(testSet).flatten()
predictionBand = pd.DataFrame(scalerBandgap.inverse_transform(predictionBand), columns = ['bandgap_energy_ev'])

#Write outputs to a file
myOutput = output_file_creator("my_submission", ID, predictionForm, predictionBand)
