
from sklearn.metrics import accuracy_score
from PIL import Image 
from sklearn.preprocessing import LabelEncoder,Normalizer
from sklearn.svm import SVC
import numpy as np
import cv2
import os

face_cascade = cv2.CascadeClassifier('/Users/kshitijaupasham/Desktop/Python/haarcascade_frontalface_default.xml')
def face_extract(filename):
    image = Image.open(filename)
    image = image.convert('RGB')
    fr = cv2.imread(filename)
    gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.001, 15)
    
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
    
        
        
trainX, trainY = load_data('/Users/kshitijaupasham/Desktop/face-recog/train/')
testX, testY = load_data('/Users/kshitijaupasham/Desktop/face-recog/val/')
np.savez_compressed('/Users/kshitijaupasham/Desktop/face-recog/faces-dataset.npz', trainX, trainY, testX, testY)

#Embedding
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

data = np.load('/Users/kshitijaupasham/Desktop/face-recog/faces-dataset.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
model = keras.models.load_model('/Users/kshitijaupasham/CV/facenet_keras.h5')
print('Loaded: ', trainX.shape, trainy.shape, testX.shape, testy.shape)

TrainX = np.asarray(embedded_return(trainX, model))
TestX = np.asarray(embedded_return(testX, model))
print(TestX.shape)
np.savez_compressed('/Users/kshitijaupasham/Desktop/face-recog/faces-embeddings.npz', TrainX, trainy, TestX, testy)
    

#model fit
data = np.load('/Users/kshitijaupasham/Desktop/face-recog/faces-embeddings.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print('Loaded: ', trainX.shape, trainy.shape, testX.shape, testy.shape)

encoder = Normalizer()
trainX = encoder.transform(trainX)
testX = encoder.transform(testX)
l_encoder = LabelEncoder()
l_encoder.fit(trainy)
trainy = l_encoder.transform(trainy)
testy = l_encoder.transform(testy)
model = SVC(kernel='linear', probability=True)
model.fit(trainX, trainy)
yhat_train = model.predict(trainX)
yhat_test = model.predict(testX)
print(accuracy_score(trainy, yhat_train))
print(accuracy_score(testy, yhat_test))
