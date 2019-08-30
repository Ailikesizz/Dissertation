import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import keras
import tensorflow as ts
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

#load model
json_file = open('/data/shangrui/ResNet50/model2_e.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = ts.keras.models.model_from_json(loaded_model_json)
#--load weights into new model
loaded_model.load_weights("/data/shangrui/ResNet50/model2_e.h5")
print("Loaded model")

X_test=pickle.load(open("/data/shangrui/bsea/X_test_512.pickle","rb"))
y_test=pickle.load(open("/data/shangrui/bsea/y_test_512.pickle","rb"))

################################################################################
pred=keras.models.Sequential.predict_proba(loaded_model,x=X_test,batch_size=8)##
################################################################################

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

