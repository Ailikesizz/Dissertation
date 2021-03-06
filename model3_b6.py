#model_f
import os
import tensorflow as tf
import keras
import tensorflow as ts
from tensorflow.keras.models import Sequential
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

#reload the data
import pickle

X_train=pickle.load(open("/data/shangrui/bsea/X_train_aug512.pickle","rb"))
y_train=pickle.load(open("/data/shangrui/bsea/y_train_aug512.pickle","rb"))

X_test=pickle.load(open("/data/shangrui/bsea/X_test_aug512.pickle","rb"))
y_test=pickle.load(open("/data/shangrui/bsea/y_test_aug512.pickle","rb"))

#X_train=X_train/255.0
#X_test=X_test/255.0


import tensorflow as tf
IMG_SHAPE = (512, 512, 3)

# Create the base model from the pre-trained model InceptionV3
#base_model = tf.keras.applications.ResNet50(input_shape=IMG_SHAPE,
                                               #include_top=False,
                                               #weights='imagenet')
#base_model.trainable=False
json_file = open("/data/shangrui/ResNet50/b3_2.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
base_model = ts.keras.models.model_from_json(loaded_model_json)
#--load weights into new model
base_model.load_weights("/data/shangrui/ResNet50/b3_2.h5")

base_model.trainable=False

print("Loaded model")
model=Sequential()
model.add(base_model.layers[0])
model.trainable=False

#extract features use base model
X_train=model.predict(X_train,batch_size=8)
X_test=model.predict(X_test,batch_size=8)

#X_train=X_train.reshape(836,(16*16*2048))
#X_test=X_test.reshape(105,(16*16*2048))


X_train=X_train.reshape(836,(16*16*2048))
X_test=X_test.reshape(105,(16*16*2048))


print('CNN done')

#SVM classifier
import sklearn
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC

#grid search for parameter C
param = [{"C": [0.01, 0.1, 1, 10, 100]}]

svm = LinearSVC(penalty='l2', loss='squared_hinge')  # As in Tang (2013)
#pro = CalibratedClassifierCV(svm)
gs= GridSearchCV(svm, param, cv=10)
clf = CalibratedClassifierCV(gs)

clf.fit(X_train,y_train)

print(clf.score(X_test,y_test))

filename = '/data/shangrui/ResNet50/b3_2_svm.sav'
pickle.dump(clf, open(filename, 'wb'))
 
# some time later...
 
#load the model from disk
#loaded_model = pickle.load(open(filename, 'rb'))
#result = loaded_model.score(X_test, Y_test)
#print(result)

