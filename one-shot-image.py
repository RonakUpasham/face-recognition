
from PIL import Image 
import numpy as np
import cv2
import os
import tensorflow
from scipy import spatial


face_cascade = cv2.CascadeClassifier('/Users/kshitijaupasham/Desktop/Python/haarcascade_frontalface_default.xml')
def face_extract(filename):
    image = Image.open(filename)
    image = image.convert('RGB')
    fr = cv2.imread(filename)
    gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        img = fr[y:y+h, x:x+w]
    img = Image.fromarray(img)
    img = img.resize((160,160))
    return np.asarray(img)


def load_data(d):
    X, Y = [],[]
    for subdir in os.listdir(d):
        p = d + subdir + '/'
        if not os.path.isdir(p):
            continue
        faces = []
        for filename in os.listdir(p):
            path = p + filename
            face = face_extract(path) 
            faces.append(face)
        l = [subdir for _ in range(len(faces))]
        X.extend(faces)
        Y.extend(l)
    return np.asarray(X), np.asarray(Y)

trainX, trainY = load_data('/Users/kshitijaupasham/Desktop/one-shot/train/')
testX, testY = load_data('/Users/kshitijaupasham/Desktop/one-shot/val/')

def embedded(model, face_pixels):
    face_pixels = face_pixels.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    samples = np.expand_dims(face_pixels, axis=0)
    yhat = model.predict(samples)
    return yhat[0]

def embedded_return(train, model):
    Train = []
    for face_pixels in train:
        Train.append(embedded(model, face_pixels))
    return Train

model = tensorflow.keras.models.load_model('/Users/kshitijaupasham/CV/facenet_keras.h5')
TrainX = np.asarray(embedded_return(trainX, model))
TestX = np.asarray(embedded_return(testX, model))

np.savez_compressed('/Users/kshitijaupasham/Desktop/one-shot/train-embeddings.npz', TrainX, trainY)
np.savez_compressed('/Users/kshitijaupasham/Desktop/one-shot/test-embeddings.npz', TestX, testY)

train_data = np.load('/Users/kshitijaupasham/Desktop/one-shot/train-embeddings.npz')
test_data = np.load('/Users/kshitijaupasham/Desktop/one-shot/test-embeddings.npz')

trainX = train_data['arr_0']
trainy = train_data['arr_1']
testX = test_data['arr_0']
testy = test_data['arr_1']

for i in range(len(testy)):
    a = testX[i]
    distance = []
    for j in range(len(trainy)):
        b = trainX[j]
        d = 1 - spatial.distance.cosine(b, a)
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

