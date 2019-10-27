import cv2

index = 0;
for n in range(1,100):
    image_path = './base_face/user.' + format(n) + '.png'
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))
    for (x, y, w, h) in faces:
        index += 1
        cv2.imwrite('head' + format(index) + '.png', image[y:y + h, x:x + w])