from __future__ import print_function
from src.network import Network
import pandas as pd
import time

print('Loading data')
df = pd.read_csv('data/creditcard.csv')
# Shuffle data
df = df.sample(frac=1).reset_index(drop=True)
split_index = int(len(df) * 0.9)

# Split data
print('Splitting data')
training_data, test_data = df[:split_index], df[split_index:]

# Convert into Numpy Arrays
train = training_data.as_matrix()
test = test_data.as_matrix()

print('Creating training & test datasets')
# Create training dataset
training_inputs = [x[:-1] for x in train]
training_results = [x[-1] for x in train]
test_inputs = [x[:-1] for x in test]
test_results = [x[-1] for x in test]
training_data = zip(training_inputs, training_results)
test_data = zip(test_inputs, test_results)
# len(training_data)

# print('Showing training data')
# print(training_data[:3])


def run(training_data, epochs, mini_batch_size, eta, test_data=None):
    # Initialize Neural Network
    print('Initializing neural network')
    nn = Network([30, 30, 1])

    print('Training network')
    start = time.time()
    nn.SGD(training_data, epochs, mini_batch_size, eta, test_data=test_data)
    print('Runtime: {}s'.format(time.time() - start))


run(training_data, 15, 5000, 20.0, test_data=test_data)
