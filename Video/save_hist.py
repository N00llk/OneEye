import cv2
import numpy as np
import matplotlib.pyplot as plt


# Для детектирования лиц используем каскады Хаара
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

path = './base_face/user.'
# Для распознавания используем локальные бинарные шаблоны
img = cv2.imread('./face_new/user.2.12.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
vals = gray.mean(axis=1).flatten()
print(vals)
# calculate histogram
counts, bins = np.histogram(vals, range(257))
# plot histogram centered on values 0..255
plt.bar(bins[:-1] - 0.5, counts, width=1, edgecolor='none')
plt.xlim([-0.5, 255.5])
plt.ylim([0, 25])
plt.savefig('hist.png', bbox_inches='tight')
plt.close()