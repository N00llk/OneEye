from pybrain3.datasets import ClassificationDataSet # Структура данных pybrain
from pybrain3.tools.shortcuts import buildNetwork
from pybrain3.supervised.trainers import BackpropTrainer
from pybrain3.structure.modules import SoftmaxLayer
from pybrain3.utilities import percentError
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

types = ["1", "2", "3", "4", "5", "6", "7", "8"]
num_fit = 44
dataset_features = np.zeros((num_fit, 80))
outputs = np.zeros((num_fit))
idx = 0
class_label = 0
for types_dir in types:
  curr_dir = os.path.join(os.path.sep, types_dir)
  all_imgs = os.listdir(os.getcwd() + curr_dir + "\\grey_ph")
  for img_file in all_imgs:
    filename = os.getcwd() + curr_dir + "\\grey_ph\\" + img_file
    img = cv2.imread(filename)
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    vals = img.mean(axis=1).flatten()
    hist = np.histogram(vals, range(40, 121))
    dataset_features[idx, :] = hist[0]

    outputs[idx] = class_label
    idx += 1
  class_label += 1

TRAIN_SIZE = 0.8 # Разделение данных на обучающую и контрольную части в пропорции 70/30%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing.data import normalize

y = outputs
X = dataset_features
X = normalize(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=TRAIN_SIZE, random_state=0, shuffle = True)


HIDDEN_NEURONS_NUM = 100 # Количество нейронов, содержащееся в скрытом слое сети
MAX_EPOCHS = 100 # Максимальное число итераций алгоритма оптимизации параметров сети

np.random.seed(0)
# Конвертация данных в структуру ClassificationDataSet
# Обучающая часть
ds_train = ClassificationDataSet(np.shape(X)[1], nb_classes=len(np.unique(y_train)))
# Первый аргумент -- количество признаков np.shape(X)[1], второй аргумент -- количество меток классов len(np.unique(y_train)))
ds_train.setField('input', X_train) # Инициализация объектов
ds_train.setField('target', y_train[:, np.newaxis]) # Инициализация ответов; np.newaxis создает вектор-столбец
ds_train._convertToOneOfMany() # Бинаризация вектора ответов
# Контрольная часть
ds_test = ClassificationDataSet(np.shape(X)[1], nb_classes=len(np.unique(y_train)))
ds_test.setField('input', X_test)
ds_test.setField('target', y_test[:, np.newaxis])
ds_test._convertToOneOfMany()


np.random.seed(0)
# Построение сети прямого распространения (Feedforward network)
net = buildNetwork(ds_train.indim, HIDDEN_NEURONS_NUM, ds_train.outdim, outclass=SoftmaxLayer)
# ds.indim -- количество нейронов входного слоя, равне количеству признаков
# ds.outdim -- количество нейронов выходного слоя, равное количеству меток классов
# SoftmaxLayer -- функция активации, пригодная для решения задачи многоклассовой классификации

init_params = np.random.random((len(net.params))) # Инициализируем веса сети для получения воспроизводимого результата
net._setParameters(init_params)

# Модуль настройки параметров pybrain использует модуль random; зафиксируем seed для получения воспроизводимого результата
np.random.seed(0)
trainer = BackpropTrainer(net, dataset=ds_train) # Инициализируем модуль оптимизации
err_train, err_val = trainer.trainUntilConvergence(maxEpochs=MAX_EPOCHS)
#line_train = plt.plot(err_train, 'b', err_val, 'r') # Построение графика
#xlab = plt.xlabel('Iterations')
#ylab = plt.ylabel('Error')
#plt.show()

res_train = net.activateOnDataset(ds_train).argmax(axis=1) # Подсчет результата на обучающей выборке
print ('Error on train: ', percentError(res_train, ds_train['target'].argmax(axis=1)), '%') # Подсчет ошибки
res_test = net.activateOnDataset(ds_test).argmax(axis=1) # Подсчет результата на тестовой выборке
print ('Error on test: ', percentError(res_test, ds_test['target'].argmax(axis=1)), '%') # Подсчет ошибки

#def plot_classification_error(hidden_neurons_num, res_train_vec, res_test_vec):
# hidden_neurons_num -- массив размера h, содержащий количество нейронов, по которому предполагается провести перебор,
#   hidden_neurons_num = [50, 100, 200, 500, 700, 1000];
# res_train_vec -- массив размера h, содержащий значения доли неправильных ответов классификации на обучении;
# res_train_vec -- массив размера h, содержащий значения доли неправильных ответов классификации на контроле
 #   plt.figure()
#    plt.plot(hidden_neurons_num, res_train_vec)
#   plt.plot(hidden_neurons_num, res_test_vec, '-r')
#    plt.show()


#hidden_neurons_num = [50, 100, 200, 500, 700, 1000]
#res_train_vec = list()
#res_test_vec = list()

#for nnum in hidden_neurons_num:
#  np.random.seed(0)
#  net = buildNetwork(ds_train.indim, nnum, ds_train.outdim, outclass=SoftmaxLayer)
#  init_params = np.random.random(
#    (len(net.params)))  # Инициализируем веса сети для получения воспроизводимого результата
#  net._setParameters(init_params)
#  trainer = BackpropTrainer(net, dataset=ds_train)
#  trainer.trainUntilConvergence(maxEpochs=MAX_EPOCHS)
#  res_train_i = net.activateOnDataset(ds_train).argmax(axis=1)
#  res_train_vec.append(percentError(res_train_i, ds_train['target'].argmax(axis=1)) / 100)
#  res_test_i = net.activateOnDataset(ds_test).argmax(axis=1)
#  res_test_vec.append(percentError(res_test_i, ds_test['target'].argmax(axis=1)) / 100)

#plot_classification_error(hidden_neurons_num, res_train_vec, res_test_vec)



#############################################################################
#Необъявленная функция предсказания. В конце надо вывести индекс максимума из массива
def predict(path):
  img = cv2.imread(path)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  vals = gray.mean(axis=1)
  hist = np.histogram(vals, range(40, 121))
  return(max(enumerate(net.activate(hist[0])),key=lambda x: x[1])[0])