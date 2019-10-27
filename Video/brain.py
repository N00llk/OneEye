import numpy
import matplotlib.pyplot

import os
import pickle
import cv2

types = ["1", "2", "3", "4"]#, "5", "6", "7", "8"]
num_fit = 120
dataset_features = numpy.zeros((num_fit, 80))
outputs = numpy.zeros((num_fit))
idx = 0
class_label = 0

for types_dir in types:
  curr_dir = os.path.join(os.path.sep, types_dir)
  all_imgs = os.listdir(os.getcwd() + curr_dir)
  for img_file in all_imgs:
    filename = os.getcwd() + curr_dir + '\\' + img_file
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    vals = gray.mean(axis=1)
    hist = numpy.histogram(vals, range(40, 121))
    dataset_features[idx, :] = hist[0]

    print(hist[0])
    outputs[idx] = class_label
    idx += 1
  class_label += 1


#with open("dataset_features.pkl", "wb") as f:
#  pickle.dump("dataset_features.pkl", f)

with open("outputs.pkl", "wb") as f:
  pickle.dump(outputs, f)


##########################################################

def sigmoid(inpt):
  return 1.0 / (1 + numpy.exp(-1 * inpt))

#f = open("dataset_features.pkl", "rb")
#data_inputs2 = pickle.load(f)
#f.close()
data_inputs2 = dataset_features

#features_STDs = numpy.std(a=data_inputs2, axis=0)
#data_inputs = data_inputs2[:, features_STDs > 50]

#f = open("outputs.pkl", "rb")
#data_outputs = pickle.load(f)
#f.close()

#HL1_neurons = 150
#input_HL1_weights = numpy.random.uniform(low=-0.1, high=0.1, size=(data_inputs.shape[1], HL1_neurons))

#HL2_neurons = 60
#HL1_HL2_weights = numpy.random.uniform(low=-0.1, high=0.1, size=(HL1_neurons, HL2_neurons))

#output_neurons = 8
#HL2_output_weights = numpy.random.uniform(low=-0.1, high=0.1, size=(HL2_neurons, output_neurons))

#H1_outputs = numpy.matmul(data_inputs[0, :], input_HL1_weights)
#H1_outputs = sigmoid(H1_outputs)
#H2_outputs = numpy.matmul(H1_outputs, HL1_HL2_weights)
#H2_outputs = sigmoid(H2_outputs)
#out_outputs = numpy.matmul(H2_outputs, HL2_output_weights)

#predicted_label = numpy.where(out_outputs == numpy.max(out_outputs))[0][0]
#print("Predicted class : ", predicted_label)


#####################################################

def relu(inpt):
  result = inpt
  result[inpt < 0] = 0
  return result

def update_weights(weights, learning_rate):
  new_weights = weights - learning_rate * weights
  return new_weights

def train_network(num_iterations, weights, data_inputs, data_outputs, learning_rate, activation="relu"):
  for iteration in range(num_iterations):
    print("Iteration ", iteration)
    for sample_idx in range(data_inputs.shape[0]):
      r1 = data_inputs[sample_idx, :]
      for idx in range(len(weights) - 1):
       curr_weights = weights[idx]
       r1 = numpy.matmul(r1, curr_weights)
       if activation == "relu":
         r1 = relu(r1)
       elif activation == "sigmoid":
         r1 = sigmoid(r1)
    curr_weights = weights[-1]
    r1 = numpy.matmul(r1, curr_weights)
    predicted_label = numpy.where(r1 == numpy.max(r1))[0][0]
    desired_label = data_outputs[sample_idx]
    if predicted_label != desired_label:
      weights = update_weights(weights, learning_rate=0.001)
  return weights

def predict_outputs(weights, data_inputs, activation="relu"):
  predictions = numpy.zeros(shape=(data_inputs.shape[0]))
  for sample_idx in range(data_inputs.shape[0]):
    r1 = data_inputs[sample_idx, :]
    for curr_weights in weights:
        r1 = numpy.matmul(r1, curr_weights)
    if activation == "relu":
        r1 = relu(r1)
    elif activation == "sigmoid":
        r1 = sigmoid(r1)
    predicted_label = numpy.where(r1 == numpy.max(r1))[0][0]
    predictions[sample_idx] = predicted_label
  return predictions

#f = open("dataset_features.pkl", "rb")
#data_inputs2 = pickle.load(f)
#f.close()

features_STDs = numpy.std(data_inputs2, axis=0)
data_inputs = data_inputs2[:, features_STDs > 0]
print(features_STDs)

f = open("outputs.pkl", "rb")
data_outputs = pickle.load(f)
f.close()

HL1_neurons = 600
input_HL1_weights = numpy.random.uniform(low=-0.1, high=0.1, size=(data_inputs.shape[1], HL1_neurons))

HL2_neurons = 300
HL1_HL2_weights = numpy.random.uniform(low=-0.1, high=0.1, size=(HL1_neurons, HL2_neurons))

output_neurons = 4
HL2_output_weights = numpy.random.uniform(low=-0.1, high=0.1, size=(HL2_neurons, output_neurons))

weights = numpy.array([input_HL1_weights, HL1_HL2_weights, HL2_output_weights])

weights = train_network(num_iterations=100, weights=weights, data_inputs=data_inputs, data_outputs=data_outputs, learning_rate=0.1, activation="relu")

predictions = predict_outputs(weights, data_inputs)
num_false = numpy.where(predictions != data_outputs)[0]
print(data_outputs)
print(predictions)
print("num_false ", num_false.size)