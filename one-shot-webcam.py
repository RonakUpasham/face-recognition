
from PIL import Image 
import numpy as np
import cv2
import os
import tensorflow
from scipy import spatial

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('/Users/kshitijaupasham/Desktop/Python/haarcascade_frontalface_default.xml')
facenet_model = tensorflow.keras.models.load_model('/Users/kshitijaupasham/CV/facenet_keras.h5')
train_embeddings = np.load('/Users/kshitijaupasham/Desktop/one-shot/train-embeddings.npz')
trainX = train_embeddings['arr_0']
trainy = train_embeddings['arr_1']

while 1:
    faces=()

    # detection
    while len(faces) == 0:
        ret, image = cap.read()
        ret1, image1 = cap.read()
        faces = face_cascade.detectMultiScale(image, 1.3, 5)
        if(len(faces) == 0):
            print('Face not found')

    # preprocessing        
    x1, y1, width, height = faces[0]
    pixels = np.asarray(Image.fromarray(image.astype('uint8'), 'RGB'))
    x2, y2 = x1 + width, y1 + height
    face = pixels[y1:y2, x1:x2]
    face_array = np.asarray(Image.fromarray(face).resize((160,160)))
    final_face = []
    final_face.append(np.asarray(face_array))
    embedding = list()

    # embeddings
    for face_pixels in final_face:
        face_pixels = face_pixels.astype('float32')
        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean) / std
        samples = np.expand_dims(face_pixels, axis=0)
        face_embedding = facenet_model.predict(samples)
        embedding.append(np.asarray(face_embedding[0]))

    # prediction
    distance = []
    for j in range(3):
        b = trainX[j]
        d = 1 - spatial.distance.cosine(b, embedding)
        distance.append(d)
    if(np.max(distance) < 0.7):
        print('Unknown face')
    else:
        if(np.argmax(distance) == 0):
            print('Person 1')
        elif(np.argmax(distance) == 1):
            print('Person 2')
        else:
            print('Person 3')

            
    cv2.imshow('webcam',image1)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()


