
import tensorflow as tf
from tensorflow.keras.datasets import boston_housing
from sklearn import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
(train_x, train_y),(test_x, test_y)=boston_housing.load_data()
train_x=preprocessing.normalize(train_x)
test_x=preprocessing.normalize(test_x)
model = Sequential([
Dense(128, activation='relu', input_shape=(train_x[0].shape)),
Dense(64, activation='relu'),
Dense(32, activation='relu'),
Dense(1)
])
model.compile(
optimizer='rmsprop',
loss='mse',
metrics=['mae']
)
history = model.fit(
x=train_x, y=train_y,
epochs=100, batch_size=1,
verbose=1,
validation_data=(test_x, test_y)
)
test_input=[[
8.65407330e-05, 0.00000000e+00, 1.13392175e-02,
0.00000000e-00, 1.12518247e-03, 1.31897603e-02,
7.53763011e-02, 1.30768051e-02, 1.09241016e-02,
4.89399752e-01, 4.41333705e-02, 8.67155186e-01,
1.75004108e-02
]]
print(
"Actual Output : 21.1",
"\nPredicted output :", model.predict(test_input)
)
