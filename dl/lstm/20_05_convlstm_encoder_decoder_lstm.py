# univariate multi-step encoder-decoder convlstm
from pathlib import Path
import time
from math import sqrt
from numpy import split
from numpy import array
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import ConvLSTM2D

#virtualenv datascience on DB900

path = Path('../../data/')

MODEL_NAME = 'LF_ConvLSTM'
period = 0.02
time_step = 0.02 / 800000.
time_vec = np.arange(0, 0.02, time_step)
f_sampling = 1 / time_step
print('Sampling Frequency = {0}/1e6 MHz'.format(f_sampling))


TOTAL_DATA = 800000

#window is a block of data, equivalent to 'week'
#ie 1000 for full; 800 if these in dataset
WINDOW = TOTAL_DATA/800 
#one tenth of data
TEST_START = TOTAL_DATA-WINDOW
#10 windows of input
N_INPUT = WINDOW*10
#number of subsequences (n steps) 
#length of each subsequence (n length) 


# split a univariate dataset into train/test sets
def split_dataset(data):
	#use last n weeks for test (enough for validation and test)
	
	train=data[:-TEST_START]
	test = data[-TEST_START:]
	# restructure into windows data
	train = array(split(train, len(train)/WINDOW))
	test = array(split(test, len(test)/WINDOW))
	return train, test
	
	
# evaluate one or more weekly forecasts against expected values
def evaluate_forecasts(actual, predicted):
	scores = list()
	# calculate an RMSE score for each day
	for i in range(actual.shape[1]):
		# calculate mse
		mse = mean_squared_error(actual[:, i], predicted[:, i])
		# calculate rmse
		rmse = sqrt(mse)
		# store
		scores.append(rmse)
	# calculate overall RMSE
	s = 0
	for row in range(actual.shape[0]):
		for col in range(actual.shape[1]):
			s += (actual[row, col] - predicted[row, col])**2
	score = sqrt(s / (actual.shape[0] * actual.shape[1]))
	return score, scores

# summarize scores
def print_scores(name, score, scores):
	#s_scores = ', '.join(['%.1f' % s for s in scores])
	#print('%s: [%.3f] %s' % ('Half hourly', score, s_scores))
	n_chunks = len(scores)
	print(type(scores))
	scores_chunked = np.array_split(scores, n_chunks)
	av_scores = []
	for chunk in scores_chunked:
	    av_scores.append(np.average(chunk))
	w_scores = ', '.join(['%.1f' % s for s in av_scores])
	print('%s: [%.3f] %s' % (name, score, w_scores))
	
# convert history into inputs and outputs

def to_supervised(train, n_input, n_out=100):

	# flatten data
	print('>>to_supervised() train.shape: {0}, n_input: {1}'.format(train.shape, n_input))
	#train.shape: (9, 80000), n_input: 80000
	data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
	X, y = list(), list()
	in_start = 0
	# step over the entire history one time step at a time
	for _ in range(len(data)):
		# define the end of the input sequence
		in_end = in_start + n_input
		out_end = in_end + n_out
		# ensure we have enough data for this instance
		if out_end < len(data):
			x_input = data[in_start:in_end, 0]
			x_input = x_input.reshape((len(x_input), 1))
			X.append(x_input)
			y.append(data[in_end:out_end, 0])
		# move along one time step
		in_start += 1
	return array(X), array(y)

# train the model
def build_model(train, n_steps, n_length, n_input):
	# prepare data
	train_x, train_y = to_supervised(train, n_input)
	# define parameters
	verbose, epochs, batch_size = 0, 20, 16
	n_features, n_outputs = train_x.shape[2], train_y.shape[1]
	# reshape into subsequences [samples, timesteps, rows, cols, channels]
	train_x = train_x.reshape((train_x.shape[0], n_steps, 1, n_length, n_features))
	# reshape output into [samples, timesteps, features]
	train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
	# define model
	model = Sequential()
	model.add(ConvLSTM2D(filters=64, kernel_size=(1,5), activation='relu', input_shape=(n_steps, 1, n_length, n_features)))
	model.add(Flatten())
	model.add(RepeatVector(n_outputs))
	model.add(LSTM(200, activation='relu', return_sequences=True))
	model.add(TimeDistributed(Dense(100, activation='relu')))
	model.add(TimeDistributed(Dense(1)))
	model.compile(loss='mse', optimizer='adam')
	# fit network
	model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
	return model

# make a forecast
def forecast(model, history, n_steps, n_length, n_input):
	# flatten data
	data = array(history)
	data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
	# retrieve last observations for input data
	input_x = data[-n_input:, 0]
	# reshape into [samples, timesteps, rows, cols, channels]
	input_x = input_x.reshape((1, n_steps, 1, n_length, 1))
	# forecast the next period
	yhat = model.predict(input_x, verbose=0)
	# we only want the vector forecast
	yhat = yhat[0]
	return yhat

# evaluate a single model
def evaluate_model(train, test, n_input, n_steps, n_length):
	# fit model
	model = build_model(train, n_steps, n_length, n_input)
	# history is a list of window data
	history = [x for x in train]
	# walk-forward validation over each period
	predictions = list()
	for i in range(len(test)):
		# predict the week
		yhat_sequence = forecast(model, history, n_steps, n_length, n_input)
		# store the predictions
		predictions.append(yhat_sequence)
		# get real observation and add to history for predicting the next week
		history.append(test[i, :])
	# evaluate predictions days for each week
	predictions = array(predictions)
	score, scores = evaluate_forecasts(test[:, :, 0], predictions)
	return score, scores

def av_scores(scores, sample_rate):
	n_chunks = len(scores) / sample_rate
	scores_chunked = np.array_split(scores, n_chunks)
	av_scores = []
	for chunk in scores_chunked:
		av_scores.append(np.mean(chunk))
	return av_scores
	
def save_obj(obj, path, name ):
	with open(path + name + '.pkl', 'wb') as f:
		pkl.dump(obj, f, pkl.HIGHEST_PROTOCOL)

def load_obj(path, name ):
	with open(path + name + '.pkl', 'rb') as f:
		return pkl.load(f)
	
def save_eval_pkl(av, scores, name):
    save_obj(av, '../../input/results/', 'forecast_averages_{0}'.format(name))
    save_obj(scores, '../../input/results/', 'forecast_scores_{0}'.format(name))


def run_forecast(train, test, dataset, name, model_name):
	# n_input is data avail for each pred
	n_input = FORECAST_DAYS * DAILY_SAMPLE_RATE
	n_steps = 10
	n_length = FORECAST_DAYS
	score, scores = evaluate_model(train, test, n_input, n_steps, n_length)
		
	# summarize scores
	print_scores('cnn', score, scores)
	av = av_scores(scores, DAILY_SAMPLE_RATE)
		
	# plot scores
	samples = list(range(100))
	plt.plot(samples, av, marker='o', label='lstm')
	plt.savefig('plots/{0}_{1}'.format( model_name, name))
	save_eval_pkl(av, scores, name+'_'+model_name)
	#plt.show()
			
def workflow(siglist = [3]):
	start = time.time()
	df_hf = pq.read_table(path/'bp_signals/train_hf_sig.parquet').to_pandas()
	for sig_id in siglist:
		print('running forecast for signal: {0}'.format(sig_id))
		sig_hf = df_hf.iloc[:, sig_id]
		train, test = split_dataset(dataset.values)
		run_forecast(sig_hf, sig_id, MODEL_NAME)
	end = time.time()
	elapsed = end - start
	print('<<workflow() for {0} signals took {1} secs'.format(len(siglist), elapsed))

if __name__ == "__main__":
	workflow()