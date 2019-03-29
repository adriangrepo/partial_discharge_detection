# univariate multi-step encoder-decoder cnn-lstm for the power usage dataset
import time
import datetime
import uuid
from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

DATE = datetime.datetime.today().strftime('%Y%m%d')
UID=str(uuid.uuid4())[:8]
MODEL_NAME = 'cnn_lstm'

data_path = '../../data/'

def read_parquet():
    xs = pq.read_pandas(data_path+'bp_signals/bad_hf_sig.parquet').to_pandas()
    sig_ids = list(xs)
    xs = xs.T
    return xs, sig_ids
    
def get_signal(sig_id, xs, roll_window=20):
    sig = xs.iloc[int(sig_id), :]
    #print('sig[:100]: {0}'.format(sig[:100]))
    s_mean = sig.rolling(window=roll_window).mean()
    s_std = sig.rolling(window=roll_window).std()
    s_mean=pd.Series(list(s_mean))
    s_std=pd.Series(list(s_std))
    df = pd.concat([sig, s_mean, s_std], axis=1)
    #print(df.loc[[100]])
    df.fillna(0, inplace=True)
    return df.values
   
# split a univariate dataset into train/validation sets
def split_dataset(data, train_start, train_stop, validation_start, validation_stop, window_len):
	#data.shape: (1442, 8)
	# split into standard weeks
	train, validation = data[train_start:train_stop], data[validation_start:validation_stop]
	# restructure into windows of weekly data
	train = array(split(train, len(train)/window_len))
	validation = array(split(validation, len(validation)/window_len))
	#train[0].shape: (7, 8)
	#validation[0].shape: (7, 8)
	return train, validation
    
def get_test(data, test_start, test_stop, window_len):
    test = data[test_start:test_stop]
    # restructure into windows of weekly data
    test = array(split(test, len(test)/window_len))
    return test

# evaluate one or more weekly forecasts against expected values
def evaluate_forecasts(actual, predicted):
	#print('evaluate_forecasts() actual.shape: {0}, predicted.shape: {1}'.format(actual.shape, predicted.shape))
	#actual.shape: (46, 7), predicted.shape: (46, 7, 1)
	# actual.shape: (1, 100), predicted.shape: (1, 10, 1)
	#actual.shape: (10, 10), predicted.shape: (10, 10, 1)
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
def summarize_scores(name, score, scores):
	s_scores = ', '.join(['%.1f' % s for s in scores])
	print('%s: [%.3f] %s' % (name, score, s_scores))

# convert history into inputs and outputs
def to_supervised(train, data_column, n_input, n_out):
	'''
	@param train: list of weeks (history)
	@param n_input: number of time steps to use as inputs
	@param n_out: number of time steps to use as outputs
	@return: returns the data in the overlapping moving window format
	train.shape: (159, 7, 8), n_input: 14, n_out: 7
	'''
	# flatten data
	#print('--to_supervised train.shape: {0}'.format(train.shape))
	data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
	#data.shape: (1113, 8)
	X, y = list(), list()
	in_start = 0
	# step over the entire history one time step at a time
	for _ in range(len(data)):
		# define the end of the input sequence
		in_end = in_start + n_input
		out_end = in_end + n_out
		# ensure we have enough data for this instance
		if out_end < len(data):
			x_input = data[in_start:in_end, data_column]
			x_input = x_input.reshape((len(x_input), 1))
			X.append(x_input)
			y.append(data[in_end:out_end, 0])
		# move along one time step
		in_start += 1
	#X.shape: (1092, 14, 1), y.shape: (1092, 7)
	return array(X), array(y)
    
def save_losses(loss, val_loss, file_name):
    np_loss = np.array(loss)
    np_val_loss = np.array(val_loss)
    arr = np.column_stack((np_loss,np_val_loss))
    columns_new = ['loss', 'val_loss']
    # pass in array and columns
    df_loss=pd.DataFrame(arr, columns=columns_new)
    df_loss.to_csv("loss/loss_{0}_{1}_{2}.txt".format(file_name, MODEL_NAME, DATE), index=False)


# train the model
def build_model(train_x, train_y, n_input, epochs, bs, kernel_size):
    # define parameters
    verbose = 0
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    #n_timesteps: 14, n_features: 1, n_outputs: 7
    	
    # reshape output into [samples, timesteps, features]
    #train_y.shape[0]: 1092, train_y.shape[1]: 7
    train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
    #train_y.shape: (1092, 7, 1)
    # define model
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=kernel_size, activation='relu', input_shape=(n_timesteps,n_features)))
    model.add(Conv1D(filters=64, kernel_size=kernel_size, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(RepeatVector(n_outputs))
    model.add(LSTM(200, dropout=0.2, recurrent_dropout=0.2, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(100, activation='relu')))
    model.add(TimeDistributed(Dense(1)))
    model.compile(loss='mse', optimizer='adam')
    # fit network
    #train_x.shape: (1092, 14, 1), train_y.shape: (1092, 7, 1)
    model.fit(train_x, train_y, epochs=epochs, batch_size=bs, verbose=verbose)
    return model

# make a forecast
def forecast(model, history, n_input):
	#len(history): 159, n_input: 14
	# flatten data
	data = array(history)
	data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
	#data.shape: (1113, 8)
	# retrieve last observations for input data
	input_x = data[-n_input:, 0]
	# reshape into [1, n_input, 1]
	input_x = input_x.reshape((1, len(input_x), 1))
	# forecast the next window
	yhat = model.predict(input_x, verbose=0)
	'''yhat[0]
		[[1393.4585]
	 [1471.1593]
	 [1532.9514]
	 [1577.6196]
	 [1594.4639]
	 [1628.5637]
	 [1673.769 ]]
 	'''
	# yhat.shape: (1, 7, 1), yhat[0].shape: (7, 1)
	# we only want the vector forecast
	yhat = yhat[0]
	return yhat
    
def forecast_test(model, test, n_input, fname):
    print('>>forecast_test() test.shape: {0}, n_input: {1}'.format(test.shape, n_input))
    history = [x for x in test]
    predictions = list()
    for i in range(len(test)):
        # predict the window
        yhat_sequence = forecast(model, history, n_input)
        # store the predictions
        predictions.append(yhat_sequence)
    predictions = array(predictions).flatten()
    np.savetxt(data_path+fname+'.out', predictions, delimiter=',')

# evaluate a single model
def evaluate_model(model, train, validation, n_input):
    print('>>evaluate_model() len(train): {0}, len(validation): {1}, n_input: {2}'.format(len(train), len(validation), n_input))
    # history is a list of weekly data
    history = [x for x in train]
    # walk-forward validation over each week
    predictions = list()
    for i in range(len(validation)):
    	# predict the window
    	yhat_sequence = forecast(model, history, n_input)
    	# store the predictions
    	predictions.append(yhat_sequence)
    	# get real observation and add to history for predicting the next window
    	history.append(validation[i, :])
    # evaluate predictions days for each week
    predictions = array(predictions)
    score, scores = evaluate_forecasts(validation[:, :, 0], predictions)
    return score, scores

def run_UCL():
	#UCL param values
	
	train_start=1
	train_stop=-328
	validation_start=-328
	validation_stop=-6
	window_len=7
	
	n_input = 14
	n_out=7
	data_column=0
	
	rel_path = '../../data/UCI/household_power_consumption/'
	# load the new file
	dataset = read_csv(rel_path+'household_power_consumption_days.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
	train, validation = split_dataset(dataset.values, train_start, train_stop, validation_start, validation_stop, window_len)
	#train.shape: (159, 7, 8), validation.shape: (46, 7, 8)
	return train, validation, data_column, n_input, n_out

def run_VSB(xs, train_start,train_stop,validation_start,validation_stop,
	window_len,data_column, sig_id):
	data = get_signal(sig_id, xs, roll_window=20)
	#data = data[:1442]
	train, validation = split_dataset(data, train_start, train_stop, validation_start, validation_stop, window_len)
	return train, validation, data

def run_forecast(train, validation, data_column, n_input, n_out, epochs=20, bs=16, kernel_size=3):
	# evaluate model and get scores
	
	# prepare data
	train_x, train_y = to_supervised(train, data_column, n_input, n_out)
	# fit model
	model = build_model(train_x, train_y, n_input, epochs, bs, kernel_size)
	score, scores = evaluate_model(model, train, validation, n_input)
	return score, scores
	
def plot_scores(n_input, n_out, scores):
	# plot scores
	pred_idxs = list(range(n_out))
	pyplot.plot(pred_idxs, scores, marker='o', label='lstm')
	pyplot.savefig('n_input_{0}_n_out_{1}'.format(n_input, n_out))
	
def epoch_sensitivity():
    #train, validation, data_column, n_input, n_out=run_UCL()
    xs, sig_ids= read_parquet()
    train_start=0
    train_stop=7000
    validation_start=7000
    validation_stop=8000
    window_len=10
    	
    n_input = 1000
    n_out=10
    data_column=0
    	
    sig_id=0
    epochs=20
    bs=16
    kernel_size=9
    train, validation=run_VSB(xs, train_start, train_stop, validation_start,validation_stop, window_len, data_column, sig_id)
    	
    model_scores=[]
    n_epochs = []
    for epochs in [10, 20, 50, 100, 200, 500]:
    	start_time = time.time()
    	score, scores=run_forecast(train, validation, data_column, n_input, n_out, epochs=epochs, bs=bs, kernel_size=kernel_size)
    	end_time=time.time()
    	# summarize scores
    	#summarize_scores('lstm', score, scores)
    	#plot_scores(n_input, n_out, scores)
    	elapsed= end_time-start_time
    	print('epochs: {0}, score: {1}, elapsed: {2} secs'.format(epochs, score, elapsed))
    	model_scores.append(score)
    	n_epochs.append(epochs)
    	
    data = {'epochs': n_epochs, 'score': model_scores}
    df = pd.DataFrame.from_dict(data)
    df.to_csv('loss/cnn_lstm_epochs_model_scores.csv', index=False)
    '''
    without dropout
    epochs: 10, score: 0.7097956531913944, elapsed: 168.32002544403076 secs
    >>evaluate_model() len(train): 700, len(validation): 100, n_input: 1000
    epochs: 20, score: 0.7301232648011526, elapsed: 327.69332909584045 secs
    >>evaluate_model() len(train): 700, len(validation): 100, n_input: 1000
    epochs: 50, score: 0.7556829912068356, elapsed: 813.7417736053467 secs
    
    with dropout:
    epochs: 10, score: 0.6937098302466647, elapsed: 174.81435537338257 secs
    >>evaluate_model() len(train): 700, len(validation): 100, n_input: 1000
    epochs: 20, score: 0.7165847589202315, elapsed: 344.6396679878235 secs
    >>evaluate_model() len(train): 700, len(validation): 100, n_input: 1000
    epochs: 50, score: 0.7636532208049757, elapsed: 853.6627323627472 secs
    '''
    
def save_model(model, name):
    #https://machinelearningmastery.com/save-load-keras-deep-learning-models/
    # serialize model to JSON
    model_json = model.to_json()
    with open(data_path+"{0}.json".format(name), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(data_path+"{0}.h5".format(name))
    print("Saved model to disk")
    
def kernel_sensitivity():
    #train, validation, data_column, n_input, n_out=run_UCL()
    xs, sig_ids= read_parquet()
    train_start=0
    train_stop=7000
    validation_start=7000
    validation_stop=8000
    window_len=10
    
    n_input = 1000
    n_out=10
    data_column=0
    
    sig_id=0
    epochs=20
    bs=16
    train, validation=run_VSB(xs, train_start, train_stop, validation_start,validation_stop, window_len, data_column, sig_id)
    
    model_scores=[]
    kernels = []
    model_n_in = []
    for n_in in [500, 1000]:
        for kernel in [3, 5, 9]:
            start_time = time.time()
            score, scores=run_forecast(train, validation, data_column, n_in, n_out, epochs=epochs, bs=bs, kernel_size=kernel)
            end_time=time.time()
            # summarize scores
            #summarize_scores('lstm', score, scores)
            #plot_scores(n_input, n_out, scores)
            elapsed= end_time-start_time
            print('n_in: {0}, kernel: {1}, score: {2}, elapsed: {3} secs'.format(n_in, kernel, score, elapsed))
            model_scores.append(score)
            kernels.append(kernel)
            model_n_in.append(n_in)
        
    
    data = {'n_input': model_n_in, 'kernel': kernels, 'score': model_scores}
    df = pd.DataFrame.from_dict(data)
    df.to_csv('loss/cnn_lstm_kernel_model_scores.csv', index=False)
	
def n_in_sensitivity():
    #train, validation, data_column, n_input, n_out=run_UCL()
    xs, sig_ids= read_parquet()
    train_start=0
    train_stop=7000
    validation_start=7000
    validation_stop=8000
    window_len=10
    	
    n_input = 1000
    n_out=10
    data_column=0
    	
    sig_id=0
    epochs=20
    bs=16
    kernel_size=3
    model_scores=[]
    model_n_in = []
    model_n_out=[]
    for n_in in [10, 50, 100, 200, 500, 1000, 2000, 5000]:
    	start_time = time.time()
    	train, validation=run_VSB(xs, train_start, train_stop, validation_start,validation_stop, window_len, data_column, sig_id)
    	
    	score, scores=run_forecast(train, validation, data_column, n_in, n_out, epochs=epochs, bs=bs, kernel_size=kernel)
    	end_time=time.time()
    	# summarize scores
    	#summarize_scores('lstm', score, scores)
    	#plot_scores(n_input, n_out, scores)
    	elapsed= end_time-start_time
    	print('n_in: {0}, score: {1}, elapsed: {2} secs'.format(n_in, score, elapsed))
    	model_scores.append(score)
    	model_n_in.append(n_in)
    	model_n_out.append(n_out)
    	
    print('model_scores: {0}'.format(model_scores))
    data = {'n_input': model_n_in, 'n_out': model_n_out, 'score': model_scores}
    df = pd.DataFrame.from_dict(data)
    df.to_csv('loss/cnn_lstm_n_in_model_scores.csv', index=False)
    #run 2
    #n_in: 10, score: 0.8154276408095572, elapsed: 101.48467946052551 secs, 1.7 mins
    #n_in: 50, score: 0.8491845131915121, elapsed: 99.90080547332764 secs
    #n_in: 100, score: 0.8392952125672757, elapsed: 105.98810935020447 secs
    #n_in: 200, score: 0.883737973585946, elapsed: 135.3347990512848 secs
    #n_in: 500, score: 0.7927576545146089, elapsed: 231.1276605129242 secs, 3.8 mins
    #n_in: 1000, score: 0.8351470918215709, elapsed: 329.08115911483765 secs, 5.5 mins
    #n_in: 2000, score: 0.9224861511493533, elapsed: 485.2296369075775 secs
    #n_in: 5000, score: 0.7581612482548865, elapsed: 456.66350960731506 secs, 7.6 mins
    	
def n_out_sensitivity():
    xs, sig_ids= read_parquet()
    train_start=0
    train_stop=7000
    validation_start=7000
    validation_stop=8000
    window_len=10
    	
    n_input = 1000
    n_out=10
    data_column=0
    	
    sig_id=0
    epochs=20
    bs=16
    kernel_size=3
    model_scores=[]
    model_n_in = []
    model_n_out=[]
    for n_out in [10, 50, 100, 200, 500]:
    	train, validation=run_VSB(xs, train_start, train_stop, validation_start,validation_stop, window_len, data_column, sig_id)
    	start_time = time.time()
    	score, scores=run_forecast(train, validation, data_column, n_in, n_out, epochs=epochs, bs=bs, kernel_size=kernel)
    	end_time=time.time()
    	# summarize scores
    	#summarize_scores('lstm', score, scores)
    	#plot_scores(n_input, n_out, scores)
    	elapsed= end_time-start_time
    	print('n_out: {0}, score: {1}, elapsed: {2} secs'.format(n_out, score, elapsed))
    	model_scores.append(score)
    	model_n_in.append(n_input)
    	model_n_out.append(n_out)
    	
    print('model_scores: {0}'.format(model_scores))
    data = {'n_input': model_n_in, 'n_out': model_n_out, 'score': model_scores}
    df = pd.DataFrame.from_dict(data)
    df.to_csv('loss/cnn_lstm_n_out_model_scores.csv', index=False)
    #n_out: 10, score: 0.7191865673236496, elapsed: 330.23196387290955 secs
    #n_out: 50, score: 1.2575179993268413, elapsed: 1109.3050870895386 secs
    #n_out: 100, score: 1.4678284209023267, elapsed: 2070.972282409668 secs
    #n_out: 200, score: 1.025708074930395, elapsed: 3932.9658987522125 secs
    #n_out: 500, score: 1.3311653790868707, elapsed: 9132.832059383392 secs
    
def gen_best_model():
    #train, validation, data_column, n_input, n_out=run_UCL()
    xs, sig_ids= read_parquet()
    train_start=0
    train_stop=7900
    validation_start=7900
    validation_stop=8000
    test_start=1000
    test_stop=8000
    window_len=10
        
    n_input = 1000
    n_out=10
    data_column=0
        
    sig_id=0
    epochs=10
    bs=16
    kernel_size=9
    train, validation, data=run_VSB(xs, train_start, train_stop, validation_start,validation_stop, window_len, data_column, sig_id)
    #pullthese out of run_forecast
    # prepare data
    train_x, train_y = to_supervised(train, data_column, n_input, n_out)
    # fit model
    model = build_model(train_x, train_y, n_input, epochs, bs, kernel_size)
    #score, scores = evaluate_model(model, train, validation, n_input)
    save_model(model, 'test_2-800k_n_in_1000_epochs_10_kernel_9')
    test=get_test(data, test_start, test_stop, window_len)
    forecast_test(model, test, n_input, fname='test_2-800k_n_in_1000_epochs_10_kernel_9')
    score, scores = evaluate_model(model, train, validation, n_input)
    print(score)
    	
if __name__ == '__main__':
	#n_in_sensitivity()
	#n_out_sensitivity()
	#epoch_sensitivity()
    #kernel_sensitivity()
    gen_best_model()
    