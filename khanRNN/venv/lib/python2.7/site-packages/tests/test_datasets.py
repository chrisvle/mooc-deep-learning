from __future__ import print_function
from keras.datasets import cifar10, cifar100

print('cifar10')
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
print(y_test[:20])

print('cifar100 fine')
(X_train, y_train), (X_test, y_test) = cifar100.load_data('fine')
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
print(y_test[:20])

print('cifar100 coarse')
(X_train, y_train), (X_test, y_test) = cifar100.load_data('coarse')
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
print(y_test[:20])


