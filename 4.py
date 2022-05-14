import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split


def get_xy(data, window):
    Y_index = np.arange(window, len(data), window)
    Y = data[Y_index]
    rows_x = len(Y)
    X = data[range(window*rows_x)]
    X = np.reshape(X, (rows_x, window, 1))
    return X, Y


def create_rnn(hidden_units, dense_units, input_shape, activation):
    model = Sequential()
    model.add(SimpleRNN(hidden_units, input_shape=input_shape, activation=activation[0], return_sequences=False))

    model.add(Dense(units=dense_units, activation=activation[1]))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error', 'mean_squared_error'])
    model.summary()
    return model


def print_error(trainY, testY, train_predict, test_predict):
    mae = (mean_absolute_error(trainY, train_predict))
    mse = (mean_squared_error(trainY, train_predict))

    print('train MAE: %.10f' % (mae))
    print('train MSE: %.10f' % (mse))

    mae = (mean_absolute_error(testY, test_predict))
    mse = (mean_squared_error(testY, test_predict))

    print('test MAE: %.10f' % (mae))
    print('test MSE: %.10f' % (mse))


def plot_result(testY, test_predict):
    actual = testY
    predictions = test_predict
    rows = len(actual)
    plt.figure(figsize=(15, 6),dpi=100)
    plt.plot(range(rows), actual, linewidth=0.9)
    plt.plot(range(rows), predictions, linewidth=0.7)
    plt.legend(['Реальная температура', 'Предсказание'])
    plt.xlabel('Номер наблюдения')
    plt.ylabel('Отмасштабированная температура')
    plt.title('Реальная и предсказанная температура')


df = pd.read_csv('temperature.csv')
df.head()

cols = df.columns
print("Выберите город: ")
for i in range(1, len(cols)-1, 1):
    print(cols[i], end=';\n')

city = input("Введите название города: ")
if city in cols:
    df = df[['datetime', city]]

df.isna().sum()
df = df.dropna()

data = df[city].values
scaler = MinMaxScaler(feature_range=(0, 1))
data = data.reshape(-1, 1)
data = scaler.fit_transform(data).flatten()


window = 20
X, y = get_xy(data, window)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = create_rnn(hidden_units=32, dense_units=1, input_shape=(window, 1), activation=['relu', 'linear'])

history = model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1, validation_split=0.2)

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

train_predict = model.predict(X_train)
test_predict = model.predict(X_test)
print_error(y_train, y_test, train_predict, test_predict)

plot_result(y_test, test_predict)