import numpy as np

# Linear Regression
X = np.array([1, 2, 3, 4], dtype=np.float32)
Y = np.array([2, 4, 6, 8], dtype=np.float32)
w = 0.0

# model output
def forward(x):
  return w * x

# loss 
def loss(y, y_pred):
  return np.mean((y_pred - y)**2)

# J = MSE = 1/N * (w*x - y)**2
# dJ/dw = 1/N * 2*(w*x - y)*(x)
def gradient(x, y, y_pred):
  return np.mean(np.dot(2*x, y_pred - y))

learning_rate = 0.01
n_iters = 100
for epoch in range(n_iters):
  y_pred = forward(X)
  l = loss(Y, y_pred)
  dw = gradient(X, Y, y_pred)
  w -= learning_rate * dw
  if epoch % 10 == 0:
    print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')

print(forward(12))
