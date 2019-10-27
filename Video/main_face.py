import cv2
import numpy as np
import os
import imutils
import matplotlib.pyplot as plt
import bs4 as bs
import urllib.request
from skimage import measure as ms
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

def insta():
    options = Options()
    options.headless = True
    browser = webdriver.Chrome('chromedriver.exe', chrome_options=options)
    browser.maximize_window()
    id_inst = input("Enter instagram login: ")
    browser.get('https://www.instagram.com/' + id_inst)
    soup = bs.BeautifulSoup(browser.page_source, 'html.parser')
    browser.close()
    index1 = 0
    index2 = 0
    for link in soup.find_all('img'):
        if link.get('srcset') != None and str(link.get('srcset'))[0] != '/':
            if not os.path.exists(id_inst):
                os.makedirs(id_inst)
                grey_path = id_inst + '/grey_ph/'
                os.makedirs(grey_path)
            urllib.request.urlretrieve(format(link.get('srcset')).split(',')[-1].split(" ")[0], id_inst +
                                       '/downloaded_'+ format(index1) + '.jpg')
            image_path = id_inst + '/downloaded_' + format(index1) + '.jpg'
            face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            image = cv2.imread(image_path)
            # print(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=30, minSize=(50, 50))
            faces_detected = "Persons found: " + format(len(faces)) # количество обнаруженных лиц на фото
            # print(faces_detected)
            for (x, y, w, h) in faces:
                cv2.imwrite(grey_path + 'grey_' + format(index2) + '.jpg',
                            cv2.cvtColor(image[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY))
                # print(id_inst + '/grey_ph/grey_' + format(index2) + '.jpg')
                index2 += 1
            index1 = index1 + 1
    if index1 == 0:
        print('Page not found')

def pars_site():
    html = urllib.request.urlopen('https://www.celeber.ru/browse')
    soup = bs.BeautifulSoup(html, 'lxml')
    index1 = 0
    index2 = 0
    for link in soup.find_all('img'):
        if link.get('src') != None:
            # print('https://www.celeber.ru' + link.get('src'))
            urllib.request.urlretrieve('https://www.celeber.ru' + link.get('src'),
                                       'sources/' + format(index1) + '.jpg')
            image_path = 'sources/' + format(index1) + '.jpg'
            face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            image = cv2.imread(image_path)
            # print(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=30, minSize=(50, 50))
            faces_detected = "Persons found: " + format(len(faces))
            # print(faces_detected)
            for (x, y, w, h) in faces:
                cv2.imwrite('grey_ph/' + format(index2) + '.jpg',
                            cv2.cvtColor(image[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY))
                index2 += 1
            index1 = index1 + 1
    print('Jobs done!') # jobs done

def web_cam_75(path_faces, face_id, name):
    cam = cv2.VideoCapture(0)
    face_detector = cv2.CascadeClassifier('C:/For_A_Reason/OneEye/Video/haarcascade_frontalface_default.xml')
    # Вводим id лица которое добавляется в имя и потом будет использовать в распознавание.
    # path_faces = input('Enter the name of the faces folder')
    # face_id = input('Enter face id end press "enter", id:')
    # name = input('Enter object name')
    if not os.path.exists(path_faces):
        os.makedirs(path_faces)
        os.makedirs(path_faces + '/photos')
        f = open(path_faces + '/' + path_faces + '.txt', 'w')
    else:
        f = open(path_faces + '/' + path_faces + '.txt', 'a')
    f.write(name + '\n')
    f.close()
    print("\nInitializing face capture. Look the camera and wait…")
    count = 0
    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            count += 1
            # Сохраняем лицо
            cv2.imwrite(path_faces + '/photos/user.' + str(face_id) + '.' + str(count) + '.jpg', gray[y:y + h, x:x + w])
        # print('Number:%', count)
        cv2.imshow('image', img)
        k = cv2.waitKey(100) & 0xff  # 'ESC'
        if k == 27:
            break
        elif count >= 75:  # выход после 75 изображений
            print("\nExiting program and cleanup stuff")
            break
    cam.release()
    cv2.destroyAllWindows()
    Study(path_faces, name)

def Study(path_faces, name):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    def getImagesAndLabels(path):
        # список файлов в папке path
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        face = []  # массив картинок
        ids = []  # id лица
        for imagePath in imagePaths:
            img = cv2.imread(imagePath)
            # перевод изображения в серый, тренер понимает только одноканальное изображение
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face.append(img)
            # id из названия
            id = int(os.path.split(imagePath)[-1].split(".")[1])
            ids.append(id)
        return face, ids

    # path = input('Enter the name of the training sample')
    # print(path_faces)
    faces, ids = getImagesAndLabels(path_faces + '/photos')
    # Тренируем train(данные, id)
    recognizer.train(faces, np.array(ids))
    # Сохраняем результат
    recognizer.write(path_faces + '/yml.yml')
    path_txt = path_faces + '/' + path_faces + '.txt'

def comp_ph(img_1, img_2):
    imageA = cv2.imread('example_compare/' + str(img_1) + '.jpg')
    imageB = cv2.imread('example_compare/' + str(img_2) + '.jpg')

    # Grayscale
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    # compute the Structural Similarity Index (SSIM) between the two
    # images, ensuring that the difference image is returned
    (score, diff) = ms.compare_ssim(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")
    # print("SSIM: {}".format(score))

    # threshold the difference image, followed by finding contours to
    # obtain the regions of the two input images that differ
    thresh = cv2.threshold(diff, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # threshold the difference image, followed by finding contours to
    # obtain the regions of the two input images that differ
    thresh = cv2.threshold(diff, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # loop over the contours
    for c in cnts:
        # compute the bounding box of the contour and then draw the
        # bounding box on both input images to represent where the two
        # images differ
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # show the output images
    cv2.imshow("Original", imageA)
    cv2.imshow("Modified", imageB)
    cv2.imshow("Diff", diff)
    cv2.imshow("Thresh", thresh)
    cv2.waitKey(0)
    return score

def id_face(path):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    # yml_id = input('Enter yml-learning id')
    recognizer.read(path + '/yml.yml')
    cascadePath = 'C:/For_A_Reason/OneEye/Video/haarcascade_frontalface_default.xml'
    faceCascade = cv2.CascadeClassifier(cascadePath);
    # Тип шрифта
    font = cv2.FONT_HERSHEY_SIMPLEX

    # iniciate id counter
    id = 0

    # Список имен для id
    names = ['None', 'Daniil Bolshakov', 'Evgen Kotov'] # face_new заполнение
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)  # set video width
    cam.set(4, 480)  # set video height

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(10, 10),
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

            # Проверяем что лицо распознано
            if (confidence < 100):
                id = names[id]
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))

            cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

        cv2.imshow('camera', img)

        k = cv2.waitKey(10) & 0xff  # 'ESC' для Выхода
        if k == 27:
            break

    cam.release()
    cv2.destroyAllWindows()

def save_gist():
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

# insta()
# pars_site()
# web_cam_100()
# print(sys.version)
# ssim_value = comp_ph(1, 2)
# print(ssim_value)
# id_face('web_face_main')
# Study('web_face_main')
web_cam_75('105', '1', 'Daniil Bolshakov')
web_cam_75('105', '2', 'Igor Shirokov')
id_face('105')
