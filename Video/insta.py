def insta():
    browser = webdriver.Chrome(r'C:\untitled\chromedriver.exe')
    browser.minimize_window()
    browser.get('https://www.instagram.com/' + str(input("ИМЯ СЮДА СУКА: ")))
    soup = bs.BeautifulSoup(browser.page_source, 'html.parser')
    browser.close()
    index1 = 0
    index2 = 0
    for link in soup.find_all('img'):
        if link.get('srcset') != None and str(link.get('srcset'))[0] != '/':
            urllib.request.urlretrieve(format(link.get('srcset')).split(',')[-1].split(" ")[0],r'C:\photos\\' + 'downloaded' + format(index1) + '.jpg')
            image_path = r'C:\photos\\' + 'downloaded' + format(index1) + '.jpg'
            face_cascade = cv2.CascadeClassifier(r'C:\Anaconda\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=30, minSize=(50, 50))
            faces_detected = "Лиц обнаружено: " + format(len(faces))
            print(faces_detected)
            for (x, y, w, h) in faces:
                cv2.imwrite(r'C:\photos\\' + 'downloaded_heads' + format(index2) + '.jpg', cv2.cvtColor(image[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY))
                index2 += 1
            index1 = index1 + 1