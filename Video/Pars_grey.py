import cv2
import bs4 as bs
import urllib.request
import cv2

html = urllib.request.urlopen('https://www.celeber.ru/browse')
soup = bs.BeautifulSoup(html, 'lxml')
index1 = 0
index2 = 0
for link in soup.find_all('img'):
    if link.get('src') != None:
        print('https://www.celeber.ru' + link.get('src'))
        urllib.request.urlretrieve('https://www.celeber.ru' + link.get('src'),
                                   'sources/' + format(index1) + '.jpg')
        image_path = 'sources/' + format(index1) + '.jpg'
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        image = cv2.imread(image_path)
        print(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=30, minSize=(50, 50))
        faces_detected = "Лиц обнаружено: " + format(len(faces))
        print(faces_detected)
        for (x, y, w, h) in faces:
            cv2.imwrite('grey_ph/' + format(index2) + '.jpg', cv2.cvtColor(image[y:y + h, x:x + w],cv2.COLOR_BGR2GRAY))
            index2 += 1
        index1 = index1 + 1