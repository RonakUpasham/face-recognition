import cv2
import numpy as np
from PIL import Image
from keras.models import load_model
from sklearn.preprocessing import Normalizer
import pickle


file='/Users/kshitijaupasham/Desktop/new-faces/ronak-test1.jpg'
model = pickle.load(open('/Users/kshitijaupasham/CV/face-recog/final_model.sav', 'rb'))
face_cascade = cv2.CascadeClassifier('/Users/kshitijaupasham/Desktop/Python/haarcascade_frontalface_default.xml')

img = cv2.imread(file)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
print(faces)

for (x1, y1, width, height) in faces:
    cv2.rectangle(img,(x1,y1),(x1+width,y1+height),(255,0,0),5)
    roi_gray = gray[y1:y1+height, x1:x1+width]
    roi_color = img[y1:y1+height, x1:x1+width]
    cv2.imwrite('/Users/kshitijaupasham/Desktop/test-face/test-img.jpg',roi_color)
    break
    
image = Image.open(file)
image = image.convert('RGB')
pixels = np.asarray(image)
x1, y1 = abs(x1), abs(y1)
x2, y2 = x1 + width, y1 + height
face = pixels[y1:y2, x1:x2]
image = Image.fromarray(face)
image = image.resize((160,160))
face_array = np.asarray(image)
new_X = []
new_X.append(face_array)
new_X = np.asarray(new_X)
new_model = load_model('/Users/kshitijaupasham/CV/facenet_keras.h5')
newTestX = list()

for face_pixels in new_X:
    face_pixels = face_pixels.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    samples = np.expand_dims(face_pixels, axis=0)
    yhat = new_model.predict(samples)
    embedding=yhat[0]
    newTestX.append(embedding)
testX = np.asarray(newTestX)
print(testX.shape)

in_encoder = Normalizer()
testX = in_encoder.transform(testX)
yhat_test = model.predict(testX)
print('Predicted: ',yhat_test)
