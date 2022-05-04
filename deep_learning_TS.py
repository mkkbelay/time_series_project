import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM, GRU
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math
import matplotlib.pyplot as plt


def get_XY(dat, time_steps):
    """
    This function has two inputs:
    - dat: this signifies a one-dimensional numpy array containing either the training or test data.
    - time_steps: signifies the number of previous time steps to use for predicting the next value of
    the series.

    Using these two inputs, the function will convert the one-dimensional array (i.e., dat) in to the
    required X and Y arrays for use in keras.
    """
    # extract indices of target array
    Y_ind = np.arange(time_steps, len(dat), time_steps)
    # use indices to extract values from the 1-d array
    Y = dat[Y_ind]
    # prepare feature set based on length of target array
    rows_x = len(Y)
    X = dat[range(time_steps*rows_x)]
    X = np.reshape(X, (rows_x, time_steps, 1))
    #return target and feature arrays
    return X, Y

def create_NN(model_type, hidden_units, dense_units, input_shape, activation, regularization):
    """
    This function is designed to return a model with a SimpleRNN and a Dense layer.
    It has four inputs that align with the parameter values for the
    SimpleRNN and Dense layers.
    """
    #create an instance of the sequential model
    model = Sequential()
    if model_type=='RNN':
        #add SimpleRNN as the first layer
        model.add(SimpleRNN(hidden_units, input_shape=input_shape,
                            activation=activation[0]))
    elif model_type=='LSTM':
        model.add(LSTM(4))
    elif model_type=='GRU':
        model.add(GRU(4))
    #add Dense model as a second layer; specify regularization option
    model.add(Dense(units=dense_units, activation=activation[1], kernel_regularizer=regularization))
    #define loss function and optimizer
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

#step 5: compute and print rmse
def print_error(trainY, testY, train_predict, test_predict):
    """
    This function outputs both the train and test RMSE.
    It has four inputs which consist of the training and test target
    values as well as the predictions.
    """
    # Error of predictions
    train_rmse = math.sqrt(mean_squared_error(trainY, train_predict))
    test_rmse = math.sqrt(mean_squared_error(testY, test_predict))
    # Print RMSE
    print('Train RMSE: %.3f RMSE' % (train_rmse))
    print('Test RMSE: %.3f RMSE' % (test_rmse))

def plot_result(trainY, testY, train_predict, test_predict):
    actual = np.append(trainY, testY)
    predictions = np.append(train_predict, test_predict)
    rows = len(actual)
    plt.figure(figsize=(15, 6), dpi=80)
    plt.plot(range(rows), actual)
    plt.plot(range(rows), predictions)
    plt.axvline(x=len(trainY), color='r')
    plt.legend(['Actual', 'Predictions'])
    plt.xlabel('Observation number after given time steps')
    plt.ylabel('Inpatients with COVID-19 (scaled)')
    plt.title('Actual and Predicted Values. The Red Line Separates The Training And Test Examples')

df=pd.read_csv('COVID-19_Reported_Patient_Impact_and_Hospital_Capacity_by_State_Timeseries.csv')

df.head()

df=df[df['state']=='TX']
df.filter(['state','date', 'percent_of_inpatients_with_covid'])
#set date column as dtype datetime and sort dataframe
df['date']=pd.to_datetime(df['date'])
df=df.sort_values(by='date')
df.head()


df=df.filter(['percent_of_inpatients_with_covid'])
df.head()
data=np.array(df.values.astype('float32'))
data

#scale data
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data).flatten()
n = len(data)
#splitting data into train and test;assign 80% of data as training
split = int(n*0.8)
train_data = data[range(split)]
test_data = data[split:]

#specify timestep
time_steps=14
#prepare training set
trainX, trainY = get_XY(train_data, time_steps)
#prepare test set
testX, testY = get_XY(test_data, time_steps)

#  call on the  function create_NN to create and train three neural network models (RNN, GRU, LSTM)
model_GRU = create_NN(model_type='GRU', hidden_units=3, dense_units=1, input_shape=(time_steps,1), activation=['tanh', 'tanh'],regularization='l1_l2')
model_RNN = create_NN(model_type='RNN', hidden_units=3, dense_units=1, input_shape=(time_steps,1), activation=['tanh', 'tanh'],regularization='l1_l2')
model_LSTM = create_NN(model_type='LSTM', hidden_units=3, dense_units=1, input_shape=(time_steps,1), activation=['tanh', 'tanh'],regularization='l1_l2')

# fit the data to the model; specify parameter values
model_GRU.fit(trainX, trainY, epochs=20, batch_size=1, verbose=2)
model_RNN.fit(trainX, trainY, epochs=20, batch_size=1, verbose=2)
model_LSTM.fit(trainX, trainY, epochs=20, batch_size=1, verbose=2)

# make predictions
#GRU
train_predict_GRU = model_GRU.predict(trainX)
test_predict_GRU = model_GRU.predict(testX)

#RNN
train_predict_RNN = model_RNN.predict(trainX)
test_predict_RNN = model_RNN.predict(testX)

#LSTM
train_predict_LSTM = model_LSTM.predict(trainX)
test_predict_LSTM = model_LSTM.predict(testX)

# Mean square error
#GRU
print_error(trainY, testY, train_predict_GRU, test_predict_GRU)

#RNN
print_error(trainY, testY, train_predict_RNN, test_predict_RNN)

#LSTM
print_error(trainY, testY, train_predict_LSTM, test_predict_LSTM)


actual = np.append(trainY, testY)
predictions_GRU = np.append(train_predict_GRU, test_predict_GRU)
predictions_RNN = np.append(train_predict_RNN, test_predict_RNN)
predictions_LSTM = np.append(train_predict_LSTM, test_predict_LSTM)
rows = len(actual)
plt.figure(figsize=(15, 6), dpi=80)
plt.plot(range(rows), actual)
plt.plot(range(rows), predictions_GRU)
plt.plot(range(rows), predictions_RNN)
plt.plot(range(rows), predictions_LSTM)
plt.axvline(x=len(trainY), color='r')
plt.legend(['Actual', 'Predictions (GRU)', 'Predictions (RNN)', 'Predictions (LSTM)'])
plt.xlabel('Observation number after given time steps')
plt.ylabel('Inpatients with COVID-19 (scaled)')
plt.title('Actual and Predicted Values. The Red Line Separates The Training And Test Examples')

plot_result(trainY, testY, train_predict, test_predict)
