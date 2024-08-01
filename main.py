import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array


gender_list = ['Male', 'Female']


# load all the pretrained models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        #the rectangle 
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        #get DAT FACE!!!!!!!!!!!!
        face = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (64, 64))
        face_array = img_to_array(face_resized)
        face_array = np.expand_dims(face_array, axis=0) / 255.0

    # result
    cv2.imshow('Gender Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
