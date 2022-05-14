import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import random
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix , classification_report, accuracy_score


def compute_evaluation_metric(model, x_test, y_test, y_predicted, y_predicted_prob):
    print("\n Accuracy Score : \n",accuracy_score(y_test,y_predicted))
    print("\n Confusion Matrix : \n ",confusion_matrix(y_test, y_predicted))
    print("\n Classification Report :\n",classification_report(y_test, y_predicted))


(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

image_number = random.randint(0, len(X_train) - 20)
plt.figure(figsize=(10, 10))
for i in range(image_number, image_number + 20):
    plt.subplot(5, 5, i-image_number + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i], cmap=plt.cm.binary)
    plt.xlabel(y_train[i])

batch_size, img_rows, img_cols = 64, 28, 28
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

model = Sequential()
model.add(Conv2D(75, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(100, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

history = model.fit(X_train, Y_train, batch_size=200, epochs=10, validation_split=0.2, verbose=2)

model.save('Model_MNIST_СNN.hdf5')

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

scores = model.evaluate(X_test, Y_test, verbose=0)
print("Test score: %f" % scores[0])
print("Test accuracy: %f" % scores[1])

y_predicted_prob = model.predict(X_test)
y_predicted = y_predicted_prob.argmax(axis = 1).astype(int)
compute_evaluation_metric(model, X_test, y_test, y_predicted, y_predicted_prob)