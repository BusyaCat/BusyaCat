import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers


def compute_evaluation_metric(model, x_test, y_test, y_predicted, y_predicted_prob):
    print("\n Accuracy Score : \n", accuracy_score(y_test, y_predicted))
    print("\n AUC Score : \n", roc_auc_score(y_test, y_predicted_prob))
    print("\n Confusion Matrix : \n ", confusion_matrix(y_test, y_predicted))
    print("\n Classification Report :\n", classification_report(y_test, y_predicted))

df = pd.read_csv('citrus.csv')
df.head()

df.describe()
df.info()

# sns.countplot(y=df['name'], data=df)
# plt.xlabel("Количество меток класса")
# plt.ylabel("Метка класса")
# plt.show()

le = LabelEncoder()
df['name'] = le.fit_transform(df['name'])

le.inverse_transform(df['name'])

X = df.drop(['name'], axis=1).values
y = df['name'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

model = keras.Sequential([
    layers.Dense(16, input_dim=X_train.shape[1], activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

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

y_predicted_prob = model.predict(X_test)
y_predicted = (y_predicted_prob > 0.5).astype(int)
compute_evaluation_metric(model, X_test, y_test, y_predicted, y_predicted_prob)
