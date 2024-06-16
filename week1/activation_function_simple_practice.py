import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

X = np.random.randn(100, 20)
y = np.random.randint(2, size=(100,1))

model = Sequential([
    Dense(64, activation='relu'),
    Dense(64, activation='tanh'),
    Dense(1, activation='sigmoid'),
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

model.fit(X, y, epochs=10)