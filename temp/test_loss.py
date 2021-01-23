import numpy as np

true = np.array([0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1])
pred = np.array([0.1, 0.065, 0.9, 0.2, 0.87, 0.67, 0.32, 0.11, 0.15, 0.01, 0.0, 0.97])
neg = np.array([1 if x == 0 else 0 for x in true])


res = np.array([0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1])

def loss_1(true, pred, s = 1e-5):
    inter = 2 * np.sum(np.abs(true*pred)) + s
    sums = np.sum(np.square(true) + np.square(pred)) + s
    return 1 - inter/sums

def loss_2(true, pred, s = 1e-5):
    inter = np.sum(true * np.square(pred)) + s
    sums = np.sum(true) + s
    return 1 - inter/sums 

def loss_3(true, pred, s = 1e-5):
    inter = np.sum(np.abs(true*pred)) + np.sum(np.square(pred)) 
    sums = np.sum(true)
    return inter/sums - 2

a = np.abs(np.random.normal(0, 1, 50))
a = a / a.max()

true = (np.abs(np.random.normal(0, 1, 50)) > 0.5) * 1
print(loss_1(true, a))
print(loss_2(true, a))

print(a)
print(true)
