"""
Deep Learning Coursework 2 Code

LSTM Recurrent Neural Network for time-series forcasting of cryptocurrency prices
"""

# import standard packages
import datetime
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import packages specifically for ML/DL
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM


# define functions
def split_and_format_data(data, seq_len):
    '''
    Split data into train(70%), validation(20%), and test(10%) sets
    Format the split sets to consist of sequences of defined length
    :param data:        data to be split and formatted
    :param seq_len:     length of sequences
    :return:            train, validation, test sets consisting of sequences
    '''
    num_samples = len(data)
    num_seq = int(num_samples / seq_len) # how many seq can be made from data (non-overlapping)

    # number of sequences for each data split
    num_seq_train = int(num_seq * 0.7)  # 70% train data
    num_seq_val = int(num_seq * 0.2)  # 20% validation data
    num_seq_test = int(num_seq * 0.1)  # 10% test data

    # determine indices for splitting data
    end_train_index = num_seq_train * seq_len
    end_val_index = end_train_index + (num_seq_val * seq_len)
    end_test_index = end_val_index + (num_seq_test * seq_len)

    # split data into sets
    train_data = data[:end_train_index]
    val_data = data[end_train_index:end_val_index]
    test_data = data[end_val_index:end_test_index]

    # organise splits and sets of sequences
    train_seq = np.array(np.split(train_data, len(train_data) / seq_len))
    val_seq = np.array(np.split(val_data, len(val_data) / seq_len))
    test_seq = np.array(np.split(test_data, len(test_data) / seq_len))

    return train_seq, val_seq, test_seq


def gen_seq_perm(data, n_input, n_out):
    '''
    Generate more data samples by including overlapping sequences of consecutive timesteps
    :param data:        data set for which to generate more samples of sequences
    :param n_input:     length of input sequence
    :param n_out:       length of output sequence
    :return:            sets of (input, output) pairs
    '''
    X, y = [], []

    in_start = 0

    for i in range(len(data)):

        # define end of input and output sequences
        in_end = in_start + n_input
        out_end = in_end + n_out

        if out_end <= len(data):

            x_seq = data[in_start:in_end,0]
            x_seq = x_seq.reshape((len(x_seq), 1))
            X.append(x_seq)

            y_seq = data[in_end:out_end, 0]
            y_seq = y_seq.reshape((len(y_seq), 1))
            y.append(y_seq)

        # increase index for start of sequence
        in_start += 1

    X = np.array(X)
    y = np.array(y)

    return X, y


def eval_forecast(y_true, y_pred):
    '''
    Calculate root mean sq error for each day and overall
    :param y_true:  actual sequential output for each input sequence
    :param y_pred:  predicted sequential output for each input sequence
    :return:        RMSE per day, RMSE overall
    '''

    # calculate RMSE for each day
    rmse_per_day = []

    for i in range(y_true.shape[1]):
        mse = mean_squared_error(y_true[:, i], y_pred[:, i])
        rmse = math.sqrt(mse)

        rmse_per_day.append(rmse)

    # calculate rmse for all samples by taking mean of rmse for each day
    final_rmse = sum(rmse_per_day)/len(rmse_per_day)

    return rmse_per_day, final_rmse


def LSTM_model_1_layer(X_train, y_train, epochs, dropout_rate, num_nodes, activation):

    # define params
    batch_size = 16

    num_tsteps = X_train.shape[1]
    num_features = X_train.shape[2]
    num_output = y_train.shape[1]

    # define model
    model = Sequential()

    model.add(LSTM(num_nodes, activation=activation, input_shape=(num_tsteps, num_features)))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(num_output))

    # compile model
    model.compile(loss='mse', optimizer='adam')

    # train model
    fit_history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    return model, fit_history


def LSTM_model_1_layer_dropout(X_train, y_train, epochs, dropout_rate, num_nodes, activation):

    # define params
    batch_size = 16

    num_tsteps = X_train.shape[1]
    num_features = X_train.shape[2]
    num_output = y_train.shape[1]

    # define model
    model = Sequential()

    model.add(LSTM(num_nodes, activation=activation, input_shape=(num_tsteps, num_features)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(num_output))

    # compile model
    model.compile(loss='mse', optimizer='adam')

    # train model
    fit_history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    return model, fit_history


def LSTM_model_2_layer(X_train, y_train, epochs, dropout_rate, num_nodes, activation):

    # define params
    batch_size = 16

    num_tsteps = X_train.shape[1]
    num_features = X_train.shape[2]
    num_output = y_train.shape[1]

    # define model
    model = Sequential()

    model.add(LSTM(num_nodes, activation=activation, input_shape=(num_tsteps, num_features), return_sequences=True))
    model.add(LSTM(num_nodes, activation=activation))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(num_output))

    # compile model
    model.compile(loss='mse', optimizer='adam')

    # train model
    fit_history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    return model, fit_history


def LSTM_model_3_layer(X_train, y_train, epochs, dropout_rate, num_nodes, activation):

    # define params
    batch_size = 16

    num_tsteps = X_train.shape[1]
    num_features = X_train.shape[2]
    num_output = y_train.shape[1]

    # define model
    model = Sequential()

    model.add(LSTM(num_nodes, activation=activation, input_shape=(num_tsteps, num_features), return_sequences=True))
    model.add(LSTM(num_nodes, activation=activation, return_sequences=True))
    model.add(LSTM(num_nodes, activation=activation))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(num_output))

    # compile model
    model.compile(loss='mse', optimizer='adam')

    # train model
    fit_history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    return model, fit_history


def forecast(model, history):
    '''
    Predict next sequence based on most recent
    :param model:       Neural network model
    :param history:     array containing all previous sequences
    :return:            prediction for next sequence
    '''
    x_input = history[-1,:,:].reshape((1,history.shape[1],history.shape[2]))   # take last seqeuce in history
    y_pred = model.predict(x_input, verbose=0)
    y_pred = y_pred[0]

    return y_pred


def test_model(model, train, test, n_input):

    # initialise history as the training data
    history = train

    # walk-forward validation over each week
    predictions = []
    for i in range(len(test)):

        # predict next week based on previous
        y_pred_seq = forecast(model, history)
        # store predicted sequence
        predictions.append(y_pred_seq)
        # add true sequence to history to predict next sequence on next pass
        history = np.append(history, [test[i, :, :]], axis=0)

    # evaluate predictions days for each week
    predictions = np.array(predictions)
    rmse_per_day, final_rmse = eval_forecast(test[:, :, 0], predictions)

    return rmse_per_day, final_rmse, predictions

def num_layers_test(train_x, train_y, val_data, n_input):

	# build and train models
    model_1_layer, model1_hist = LSTM_model_1_layer(train_x, train_y, epochs=20, dropout_rate=0, num_nodes=100, activation='relu')
    model_2_layer, model2_hist = LSTM_model_2_layer(train_x, train_y, epochs=20, dropout_rate=0, num_nodes=100, activation='relu')
    model_3_layer, model3_hist = LSTM_model_3_layer(train_x, train_y, epochs=20, dropout_rate=0, num_nodes=100, activation='relu')

	# use walk-forward validation to get prediction for every week in validation set for each model
    rmse_per_day1, model_rmse1, predictions1 = test_model(model_1_layer, train_data, val_data, n_input)
    rmse_per_day2, model_rmse2, predictions2 = test_model(model_2_layer, train_data, val_data, n_input)
    rmse_per_day3, model_rmse3, predictions3 = test_model(model_3_layer, train_data, val_data, n_input)

    print('1 layer LSTM model RMSE per day:    ', str(rmse_per_day1))
    print('2 layer LSTM model RMSE per day:    ', str(rmse_per_day2))
    print('3 layer LSTM model RMSE per day:    ', str(rmse_per_day3), '\n')

    print('1 layer LSTM model RMSE total:    ', str(model_rmse1))
    print('2 layer LSTM model RMSE total:    ', str(model_rmse2))
    print('3 layer LSTM model RMSE total:    ', str(model_rmse3), '\n')

    # plot rmse for each day in output sequence
    plt.plot(rmse_per_day1, marker='o', label='1 LSTM layer')
    plt.plot(rmse_per_day2, marker='o', label='2 LSTM layers')
    plt.plot(rmse_per_day3, marker='o', label='3 LSTM layers')
    plt.title('Root mean squared error per day in predicted output sequence')
    plt.xlabel('Day')
    plt.ylabel('RMSE')
    plt.legend()
    plt.show()

	# plot predictions - final test weeks
    final_y_pred1 = predictions1[-1]
    final_y_pred2 = predictions2[-1]
    final_y_pred3 = predictions3[-1]
    final_val = val_data[-1]

    days = np.arange(1,8)
    plt.plot(days, final_y_pred1, label='1 LSTM layer model prediction')
    plt.plot(days, final_y_pred2, label='2 LSTM layers model prediction')
    plt.plot(days, final_y_pred3, label='3 LSTM layers model prediction')
    plt.plot(days, final_val, label='actual values')
    plt.title('Price vs time for BTC for last week in validation set')
    plt.xlabel('Day')  # needs sorted
    plt.ylabel('Closing Price, $')
    plt.legend()
    plt.show()

    # plot predictions - all test weeks
    y_pred_flat1 = predictions1.reshape((predictions1.shape[0] * predictions1.shape[1]))
    y_pred_flat2 = predictions2.reshape((predictions2.shape[0] * predictions2.shape[1]))
    y_pred_flat3 = predictions3.reshape((predictions3.shape[0] * predictions3.shape[1]))
    val_data_flat = val_data.reshape((val_data.shape[0] * val_data.shape[1]))

    plt.plot(y_pred_flat1, label='1 LSTM layer model predictions')
    plt.plot(y_pred_flat2, label='2 LSTM layers model predictions')
    plt.plot(y_pred_flat3, label='3 LSTM layers model predictions')
    plt.plot(val_data_flat, label='actual values')
    plt.title('Price vs time for BTC for all weeks in validation set')
    plt.xlabel('Day')  # needs sorted
    plt.ylabel('Closing Price, $')
    plt.legend()
    plt.show()

    # plot loss vs epoch for each model
    model1_loss = model1_hist.history['loss']
    model2_loss = model2_hist.history['loss']
    model3_loss = model3_hist.history['loss']

    plt.plot(model1_loss, label='Training loss (1 LSTM layer)')
    plt.plot(model2_loss, label='Training loss (2 LSTM layers)')
    plt.plot(model3_loss, label='Training loss (3 LSTM layers)')
    plt.title("Training Loss vs epoch")
    plt.xlabel('Epoch')  # needs sorted
    plt.ylabel('Loss: Mean Squared Error')
    plt.legend()
    plt.show()

def num_nodes_test(train_x, train_y, val_data, n_input):

	# build and train models
    model_1, model1_hist = LSTM_model_1_layer(train_x, train_y, epochs=20, dropout_rate=0, num_nodes=50, activation='relu')
    model_2, model2_hist = LSTM_model_1_layer(train_x, train_y, epochs=20, dropout_rate=0, num_nodes=100, activation='relu')
    model_3, model3_hist = LSTM_model_1_layer(train_x, train_y, epochs=20, dropout_rate=0, num_nodes=200, activation='relu')

	# use walk-forward validation to get prediction for every week in validation set for each model
    rmse_per_day1, model_rmse1, predictions1 = test_model(model_1, train_data, val_data, n_input)
    rmse_per_day2, model_rmse2, predictions2 = test_model(model_2, train_data, val_data, n_input)
    rmse_per_day3, model_rmse3, predictions3 = test_model(model_3, train_data, val_data, n_input)

    print('50 nodes LSTM model RMSE per day:    ', str(rmse_per_day1))
    print('100 nodes LSTM model RMSE per day:    ', str(rmse_per_day2))
    print('200 nodes LSTM model RMSE per day:    ', str(rmse_per_day3), '\n')

    print('50 nodes LSTM model RMSE total:    ', str(model_rmse1))
    print('100 nodes LSTM model RMSE total:    ', str(model_rmse2))
    print('200 nodes LSTM model RMSE total:    ', str(model_rmse3), '\n')

    days = np.arange(1, 8)
    # plot rmse for each day in output sequence
    plt.plot(days, rmse_per_day1, marker='o', label='50 nodes')
    plt.plot(days, rmse_per_day2, marker='o', label='100 nodes')
    plt.plot(days, rmse_per_day3, marker='o', label='200 nodes')
    plt.title('Root mean squared error per day in predicted output sequence')
    plt.xlabel('Day')
    plt.ylabel('RMSE')
    plt.legend()
    plt.show()

	# plot predictions - final test weeks
    final_y_pred1 = predictions1[-1]
    final_y_pred2 = predictions2[-1]
    final_y_pred3 = predictions3[-1]
    final_val = val_data[-1]

    plt.plot(days, final_y_pred1, label='50 nodes model prediction')
    plt.plot(days, final_y_pred2, label='100 nodes model prediction')
    plt.plot(days, final_y_pred3, label='200 nodes model prediction')
    plt.plot(days, final_val, label='actual values')
    plt.title('Price vs time for BTC for last week in validation set')
    plt.xlabel('Day')  # needs sorted
    plt.ylabel('Closing Price, $')
    plt.legend()
    plt.show()

    # plot predictions - all test weeks
    y_pred_flat1 = predictions1.reshape((predictions1.shape[0] * predictions1.shape[1]))
    y_pred_flat2 = predictions2.reshape((predictions2.shape[0] * predictions2.shape[1]))
    y_pred_flat3 = predictions3.reshape((predictions3.shape[0] * predictions3.shape[1]))
    val_data_flat = val_data.reshape((val_data.shape[0] * val_data.shape[1]))

    plt.plot(y_pred_flat1, label='50 nodes model predictions')
    plt.plot(y_pred_flat2, label='100 nodes model predictions')
    plt.plot(y_pred_flat3, label='200 nodes model predictions')
    plt.plot(val_data_flat, label='actual values')
    plt.title('Price vs time for BTC for all weeks in validation set')
    plt.xlabel('Day')  # needs sorted
    plt.ylabel('Closing Price, $')
    plt.legend()
    plt.show()

    # plot loss vs epoch for each model
    model1_loss = model1_hist.history['loss']
    model2_loss = model2_hist.history['loss']
    model3_loss = model3_hist.history['loss']

    plt.plot(model1_loss, label='Training loss (50 nodes)')
    plt.plot(model2_loss, label='Training loss (100 nodes)')
    plt.plot(model3_loss, label='Training loss (200 nodes)')
    plt.title("Training Loss vs epoch")
    plt.xlabel('Epoch')  # needs sorted
    plt.ylabel('Loss: Mean Squared Error')
    plt.legend()
    plt.show()

def activation_test(train_x, train_y, val_data, n_input):

	# build and train models
    model_1, model1_hist = LSTM_model_1_layer(train_x, train_y, epochs=20, dropout_rate=0, num_nodes=100, activation='relu')
    model_2, model2_hist = LSTM_model_1_layer(train_x, train_y, epochs=20, dropout_rate=0, num_nodes=100, activation='linear')

	# use walk-forward validation to get prediction for every week in validation set for each model
    rmse_per_day1, model_rmse1, predictions1 = test_model(model_1, train_data, val_data, n_input)
    rmse_per_day2, model_rmse2, predictions2 = test_model(model_2, train_data, val_data, n_input)

    print('relu activation function LSTM model RMSE per day:    ', str(rmse_per_day1))
    print('linear activation function LSTM model RMSE per day:    ', str(rmse_per_day2))

    print('relu activation function LSTM model RMSE total:    ', str(model_rmse1))
    print('linear activation function LSTM model RMSE total:    ', str(model_rmse2))


    days = np.arange(1,8)
    # plot rmse for each day in output sequence
    plt.plot(days, rmse_per_day1, marker='o', label='relu')
    plt.plot(days, rmse_per_day2, marker='o', label='linear')
    plt.title('Root mean squared error per day in predicted output sequence')
    plt.xlabel('Day')
    plt.ylabel('RMSE')
    plt.legend()
    plt.show()

	# plot predictions - final test weeks
    final_y_pred1 = predictions1[-1]
    final_y_pred2 = predictions2[-1]
    final_val = val_data[-1]

    plt.plot(days, final_y_pred1, label='relu activation prediction')
    plt.plot(days, final_y_pred2, label='linear activation prediction')
    plt.plot(days, final_val, label='actual values')
    plt.title('Price vs time for BTC for last week in validation set')
    plt.xlabel('Day')  # needs sorted
    plt.ylabel('Closing Price, $')
    plt.legend()
    plt.show()

    # plot predictions - all test weeks
    y_pred_flat1 = predictions1.reshape((predictions1.shape[0] * predictions1.shape[1]))
    y_pred_flat2 = predictions2.reshape((predictions2.shape[0] * predictions2.shape[1]))
    val_data_flat = val_data.reshape((val_data.shape[0] * val_data.shape[1]))

    plt.plot(y_pred_flat1, label='relu activation predictions')
    plt.plot(y_pred_flat2, label='linear activation predictions')
    plt.plot(val_data_flat, label='actual values')
    plt.title('Price vs time for BTC for all weeks in validation set')
    plt.xlabel('Day')  # needs sorted
    plt.ylabel('Closing Price, $')
    plt.legend()
    plt.show()

    # plot loss vs epoch for each model
    model1_loss = model1_hist.history['loss']
    model2_loss = model2_hist.history['loss']

    plt.plot(model1_loss, label='Training loss (relu)')
    plt.plot(model2_loss, label='Training loss (linear)')
    plt.title("Training Loss vs epoch")
    plt.xlabel('Epoch')  # needs sorted
    plt.ylabel('Loss: Mean Squared Error')
    plt.legend()
    plt.show()

def dropout_test(train_x, train_y, val_data, n_input):

	# build and train models
    model_1, model1_hist = LSTM_model_1_layer(train_x, train_y, epochs=20, dropout_rate=0, num_nodes=100, activation='relu')
    model_2, model2_hist = LSTM_model_1_layer_dropout(train_x, train_y, epochs=20, dropout_rate=0.2, num_nodes=100, activation='relu')
    model_3, model3_hist = LSTM_model_1_layer_dropout(train_x, train_y, epochs=20, dropout_rate=0.4, num_nodes=100, activation='relu')

	# use walk-forward validation to get prediction for every week in validation set for each model
    rmse_per_day1, model_rmse1, predictions1 = test_model(model_1, train_data, val_data, n_input)
    rmse_per_day2, model_rmse2, predictions2 = test_model(model_2, train_data, val_data, n_input)
    rmse_per_day3, model_rmse3, predictions3 = test_model(model_3, train_data, val_data, n_input)

    print('LSTM model 1 (dropout_rate=0) RMSE per day:    ', str(rmse_per_day1))
    print('LSTM model 2 (dropout_rate=0.2) RMSE per day:    ', str(rmse_per_day2))
    print('LSTM model 3 (dropout_rate=0.4) RMSE per day:    ', str(rmse_per_day3), '\n')

    print('LSTM model 1 (dropout_rate=0) RMSE total:    ', str(model_rmse1))
    print('LSTM model 2 (dropout_rate=0.2) RMSE total:    ', str(model_rmse2))
    print('LSTM model 3 (dropout_rate=0.4) RMSE total:    ', str(model_rmse3), '\n')

    days = np.arange(1,8)
    # plot rmse for each day in output sequence
    plt.plot(days, rmse_per_day1, marker='o', label='dropout_rate=0')
    plt.plot(days, rmse_per_day2, marker='o', label='dropout_rate=0.2')
    plt.plot(days, rmse_per_day3, marker='o', label='dropout_rate=0.4')
    plt.title('Root mean squared error per day in predicted output sequence')
    plt.xlabel('Day')
    plt.ylabel('RMSE')
    plt.legend()
    plt.show()

	# plot predictions - final test weeks
    final_y_pred1 = predictions1[-1]
    final_y_pred2 = predictions2[-1]
    final_y_pred3 = predictions3[-1]
    final_val = val_data[-1]

    plt.plot(days, final_y_pred1, label='dropout_rate=0 prediction')
    plt.plot(days, final_y_pred2, label='dropout_rate=0.2 prediction')
    plt.plot(days, final_y_pred3, label='dropout_rate=0.4 prediction')
    plt.plot(days, final_val, label='actual values')
    plt.title('Price vs time for BTC for last week in validation set')
    plt.xlabel('Day')  # needs sorted
    plt.ylabel('Closing Price, $')
    plt.legend()
    plt.show()

    # plot predictions - all test weeks
    y_pred_flat1 = predictions1.reshape((predictions1.shape[0] * predictions1.shape[1]))
    y_pred_flat2 = predictions2.reshape((predictions2.shape[0] * predictions2.shape[1]))
    y_pred_flat3 = predictions3.reshape((predictions3.shape[0] * predictions3.shape[1]))
    val_data_flat = val_data.reshape((val_data.shape[0] * val_data.shape[1]))

    plt.plot(y_pred_flat1, label='dropout_rate=0 predictions')
    plt.plot(y_pred_flat2, label='dropout_rate=0.2 predictions')
    plt.plot(y_pred_flat3, label='dropout_rate=0.4 predictions')
    plt.plot(val_data_flat, label='actual values')
    plt.title('Price vs time for BTC for all weeks in validation set')
    plt.xlabel('Day')  # needs sorted
    plt.ylabel('Closing Price, $')
    plt.legend()
    plt.show()

    # plot loss vs epoch for each model
    model1_loss = model1_hist.history['loss']
    model2_loss = model2_hist.history['loss']
    model3_loss = model3_hist.history['loss']

    plt.plot(model1_loss, label='Training loss (dropout_rate=0)')
    plt.plot(model2_loss, label='Training loss (dropout_rate=0.2)')
    plt.plot(model3_loss, label='Training loss (dropout_rate=0.4)')
    plt.title("Training Loss vs epoch")
    plt.xlabel('Epoch')  # needs sorted
    plt.ylabel('Loss: Mean Squared Error')
    plt.legend()
    plt.show()

def epoch_test(train_x, train_y, val_data, n_input):

	# build and train models
    model_1, model1_hist = LSTM_model_1_layer(train_x, train_y, epochs=20, dropout_rate=0, num_nodes=100, activation='relu')
    model_2, model2_hist = LSTM_model_1_layer(train_x, train_y, epochs=40, dropout_rate=0, num_nodes=100, activation='relu')
    model_3, model3_hist = LSTM_model_1_layer(train_x, train_y, epochs=60, dropout_rate=0, num_nodes=100, activation='relu')

	# use walk-forward validation to get prediction for every week in validation set for each model
    rmse_per_day1, model_rmse1, predictions1 = test_model(model_1, train_data, val_data, n_input)
    rmse_per_day2, model_rmse2, predictions2 = test_model(model_2, train_data, val_data, n_input)
    rmse_per_day3, model_rmse3, predictions3 = test_model(model_3, train_data, val_data, n_input)

    print('LSTM model 1 (epochs=20) RMSE per day:    ', str(rmse_per_day1))
    print('LSTM model 2 (epochs=40) RMSE per day:    ', str(rmse_per_day2))
    print('LSTM model 3 (epochs=60) RMSE per day:    ', str(rmse_per_day3), '\n')

    print('LSTM model 1 (epochs=20) RMSE total:    ', str(model_rmse1))
    print('LSTM model 2 (epochs=40) RMSE total:    ', str(model_rmse2))
    print('LSTM model 3 (epochs=60) RMSE total:    ', str(model_rmse3), '\n')

    days = np.arange(1,8)
    # plot rmse for each day in output sequence
    plt.plot(days, rmse_per_day1, marker='o', label='epochs=20')
    plt.plot(days, rmse_per_day2, marker='o', label='epochs=40')
    plt.plot(days, rmse_per_day3, marker='o', label='epochs=60')
    plt.title('Root mean squared error per day in predicted output sequence')
    plt.xlabel('Day')
    plt.ylabel('RMSE')
    plt.legend()
    plt.show()

	# plot predictions - final test weeks
    final_y_pred1 = predictions1[-1]
    final_y_pred2 = predictions2[-1]
    final_y_pred3 = predictions3[-1]
    final_val = val_data[-1]

    plt.plot(days, final_y_pred1, marker='o', label='epochs=20 prediction')
    plt.plot(days, final_y_pred2, marker='o', label='epochs=40 prediction')
    plt.plot(days, final_y_pred3, marker='o', label='epochs=60 prediction')
    plt.plot(days, final_val, marker='o', label='actual values')
    plt.title('Price vs time for BTC for last week in validation set')
    plt.xlabel('Day')  # needs sorted
    plt.ylabel('Closing Price, $')
    plt.legend()
    plt.show()

    # plot predictions - all test weeks
    y_pred_flat1 = predictions1.reshape((predictions1.shape[0] * predictions1.shape[1]))
    y_pred_flat2 = predictions2.reshape((predictions2.shape[0] * predictions2.shape[1]))
    y_pred_flat3 = predictions3.reshape((predictions3.shape[0] * predictions3.shape[1]))
    val_data_flat = val_data.reshape((val_data.shape[0] * val_data.shape[1]))

    plt.plot(y_pred_flat1, marker='o', label='epochs=20 predictions')
    plt.plot(y_pred_flat2, marker='o', label='epochs=40 predictions')
    plt.plot(y_pred_flat3, marker='o', label='epochs=60 predictions')
    plt.plot(val_data_flat, marker='o', label='actual values')
    plt.title('Price vs time for BTC for all weeks in validation set')
    plt.xlabel('Day')  # needs sorted
    plt.ylabel('Closing Price, $')
    plt.legend()
    plt.show()

    # plot loss vs epoch for each model
    model1_loss = model1_hist.history['loss']
    model2_loss = model2_hist.history['loss']
    model3_loss = model3_hist.history['loss']

    plt.plot(model1_loss, marker='o', label='Training loss (epochs=20)')
    plt.plot(model2_loss, marker='o', label='Training loss (epochs=40)')
    plt.plot(model3_loss, marker='o', label='Training loss (epochs=60)')
    plt.title("Training Loss vs epoch")
    plt.xlabel('Epoch')  # needs sorted
    plt.ylabel('Loss: Mean Squared Error')
    plt.legend()
    plt.show()

def final_model(train_x, train_y, test_data, n_input):

	# build and train models
    #model, model_hist = LSTM_model_2_layer(train_x, train_y, epochs=50, dropout_rate=0.2, num_nodes=50, activation='relu')
    model, model_hist = LSTM_model_1_layer(train_x, train_y, epochs=40, dropout_rate=0, num_nodes=100, activation='relu')

	# use walk-forward validation to get prediction for every week in validation set for each model
    rmse_per_day, model_rmse, predictions = test_model(model, val_data, test_data, n_input)

    print('Final LSTM model RMSE per day:    ', str(rmse_per_day), '\n')

    print('Final LSTM model RMSE total:    ', str(model_rmse), '\n')

    days = np.arange(1,8)
    # plot rmse for each day in output sequence
    plt.plot(days, rmse_per_day, marker='o')
    plt.title('Root mean squared error per day in predicted output sequence')
    plt.xlabel('Day')
    plt.ylabel('RMSE')
    plt.show()

	# plot predictions - final test weeks
    final_y_pred = predictions[-1]
    final_test = test_data[-1]

    plt.plot(days, final_y_pred, marker='o', label='predicted values')
    plt.plot(days, final_test, marker='o', label='actual values')
    plt.title('Price vs time for BTC for last week in test set')
    plt.xlabel('Day')  # needs sorted
    plt.ylabel('Closing Price, $')
    plt.legend()
    plt.show()

    # plot predictions - all test weeks
    y_pred_flat = predictions.reshape((predictions.shape[0] * predictions.shape[1]))
    test_data_flat = test_data.reshape((test_data.shape[0] * test_data.shape[1]))

    plt.plot(y_pred_flat, label='predicted values')
    plt.plot(test_data_flat, label='actual values')
    plt.title('Price vs time for BTC for all weeks in validation set')
    plt.xlabel('Day')  # needs sorted
    plt.ylabel('Closing Price, $')
    plt.legend()
    plt.show()

    # plot loss vs epoch for each model
    model_loss = model_hist.history['loss']

    plt.plot(model_loss)
    plt.title("Training Loss vs epoch")
    plt.xlabel('Epoch')  # needs sorted
    plt.ylabel('Loss: Mean Squared Error')
    plt.show()

# import data as pandas dataframe
csv_path = "https://raw.githubusercontent.com/curiousily/Deep-Learning-For-Hackers/master/data/3.stock-prediction/BTC-USD.csv"
BTC_data = pd.read_csv(csv_path)

# inspect data
#print(BTC_data.head())
#print(BTC_data.isnull().values.sum())    # check for missing values
date_df = pd.to_datetime(BTC_data.pop('Date'), format='%Y.%m.%d')    # remove date from dataframe and change datatype
BTC_data.set_index(date_df, inplace=True)   # set dates as index
BTC_data.drop(columns=['Open', 'High', 'Low', 'Adj Close', 'Volume'], inplace=True)    # drop columns - just take close price
#print(BTC_data.head())

'''
Main block of code 
comment/uncomment lines to run tests or final model 
'''

# set params
n_input = 7
n_out = 7
seq_len = 7

# set train/test data
train_data, val_data, test_data = split_and_format_data(BTC_data.values, seq_len)

# construct more training data by getting permutations of sequences
train_x, train_y = gen_seq_perm(BTC_data.values, n_input, n_out)

# Test performance for different numbers of LSTM layers
# num_layers_test(train_x, train_y, val_data, n_input)

# Test performance for different numbers of LSTM nodes per layer
# num_nodes_test(train_x, train_y, val_data, n_input)

# Test performance for different activation functions
# activation_test(train_x, train_y, val_data, n_input)

# Test performance for different dropout rate values
# dropout_test(train_x, train_y, val_data, n_input)

# Test performance for different number epochs
# epoch_test(train_x, train_y, val_data, n_input)

# final model based on hyperparam tests
final_model(train_x, train_y, test_data, n_input)




