import cv2
import numpy as np
from PIL import Image
from keras.models import load_model
from sklearn.preprocessing import Normalizer
import pickle

cap = cv2.VideoCapture(0)
classifier = pickle.load(open('/Users/kshitijaupasham/CV/face-recog/final_model.sav', 'rb'))
face_cascade = cv2.CascadeClassifier('/Users/kshitijaupasham/Desktop/Python/haarcascade_frontalface_default.xml')
facenet_model = load_model('/Users/kshitijaupasham/CV/facenet_keras.h5')

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
    normalizer = Normalizer()
    embedding = normalizer.transform(embedding)
    face_predicted = classifier.predict(embedding)
    print('Predicted: ',face_predicted)
    cv2.imshow('webcam',image1)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
