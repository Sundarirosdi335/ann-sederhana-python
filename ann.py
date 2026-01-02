import numpy as np

# Data latih
X = np.array([[40], [55], [60], [70], [85]])
y = np.array([[0], [0], [1], [1], [1]])

# Inisialisasi bobot
weight = np.random.rand()
bias = np.random.rand()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

learning_rate = 0.1
epochs = 1000

for _ in range(epochs):
    for i in range(len(X)):
        z = X[i] * weight + bias
        output = sigmoid(z)
        error = y[i] - output
        weight += learning_rate * error * X[i]
        bias += learning_rate * error

# Testing
nilai_uji = 65
z = nilai_uji * weight + bias
hasil = sigmoid(z)

print("Nilai:", nilai_uji)
print("Output ANN:", hasil)
print("Hasil:", "LULUS" if hasil >= 0.5 else "TIDAK LULUS")
