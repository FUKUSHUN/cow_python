import keras.utils
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
import numpy as np
import pandas as pd
import pickle

filename = "training_data.csv"
data_set = pd.read_csv(filename, sep = ",", header = None, usecols = [2,3,5,6,8,14,15], names=('RTime', 'WTime', 'Velocity', 'MVelocity', 'Distance', 'Target1', 'Target2'))
train_data_set1 = data_set.drop("Target2", axis = 1)
train_data_set2 = data_set.drop("Target1", axis = 1)

x = pd.DataFrame(train_data_set1.drop("Target1", axis = 1))
y = pd.DataFrame(train_data_set1["Target1"])

w = pd.DataFrame(train_data_set2.drop("Target2", axis = 1))
z = pd.DataFrame(train_data_set2["Target2"])

#説明変数・目的変数をそれぞれ訓練データ・テストデータに分割
X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=0.10)
W_train, W_test, Z_train, Z_test = train_test_split(w,z,test_size=0.10)


#データの整形
X_train = X_train.astype(np.float)
X_test = X_test.astype(np.float)

Y_train = keras.utils.to_categorical(Y_train, 2)
Y_test = keras.utils.to_categorical(Y_test, 2)

W_train = W_train.astype(np.float)
W_test = W_test.astype(np.float)

Z_train = keras.utils.to_categorical(Z_train, 3)
Z_test = keras.utils.to_categorical(Z_test, 3)


# Model setting
n_in =  5
n_hidden = 3
n_out = 2

model = Sequential()
model.add(Dense(n_hidden, input_dim = n_in))
model.add(Activation('tanh'))

model.add(Dense(n_hidden))
model.add(Activation('tanh'))

model.add(Dense(n_out))
model.add(Activation('softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer=SGD(lr = 0.01), metrics = ['accuracy'])

# Model setting
n_in =  5
n_hidden = 3
n_out = 3

model2 = Sequential()
model2.add(Dense(n_hidden, input_dim = n_in))
model2.add(Activation('tanh'))

model2.add(Dense(n_hidden))
model2.add(Activation('tanh'))

model2.add(Dense(n_out))
model2.add(Activation('softmax'))

model2.compile(loss = 'categorical_crossentropy', optimizer=SGD(lr = 0.01), metrics = ['accuracy'])


# Model learing
epochs = 120
batch_size = 50
model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size)

epochs = 120
batch_size = 50
model2.fit(W_train, Z_train, epochs=epochs, batch_size=batch_size)


# Verify the varidity
loss_and_metrics = model.evaluate(X_test, Y_test)
loss_and_metrics2 = model2.evaluate(W_test, Z_test)
print(loss_and_metrics)
print(loss_and_metrics2)


# モデルを保存する
filename1 = 'mp/model.pickle'
pickle.dump(model, open(filename1, 'wb'))
filename2 = 'mp/model2.pickle'
pickle.dump(model2, open(filename2, 'wb'))