import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
import keras
import tensorflow as ts
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
IMG_SHAPE = (512, 512, 3)
import pickle
filename = '/data/shangrui/ResNet50/b3_2_svm.sav'
#pickle.dump(clf, open(filename, 'wb'))
X_test=pickle.load(open("/data/shangrui/bsea/X_test_aug512.pickle","rb"))

json_file = open("/data/shangrui/ResNet50/b3_2.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
base_model = ts.keras.models.model_from_json(loaded_model_json)
#--load weights into new model
base_model.load_weights("/data/shangrui/ResNet50/b3_2.h5")

print("Loaded model")
model=Sequential()
model.add(base_model.layers[0])
model.trainable=False

X_test=model.predict(X_test)

y_test=pickle.load(open("/data/shangrui/bsea/y_test_aug512.pickle","rb")) 

X_test=X_test.reshape(105,(16*16*2048))
#load the model from disk

loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)
print(result)

pred=loaded_model.predict_proba(X_test)

pred=pd.DataFrame(pred)
#pred.to_csv('/home/shangrui/pred.csv')

y_pred=[]
y_true=[]
for i in range(len(pred)):
  pro=np.amax(pred.iloc[i])
  if pro>0.9:
    index=pred.columns[(pred == pro).iloc[i]]
    index = index.to_list()
    index=index[0]
    y_pred.append(index)
    y_true.append(y_test[i])
    
print(len(y_true))
print(len(y_true)/len(y_test))

acc=accuracy_score(y_true, y_pred)
print(acc)
