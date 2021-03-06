import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import matplotlib.pyplot as plt

def plot_lc(model,name): 

  plt.figure() 

  plt.plot(model.history[name]) 

  plt.plot(model.history['val_'+name]) 

  plt.title('Learning Curve') 

  plt.ylabel(name) 

  plt.xlabel('Epoch') 

  plt.legend(['Train', 'Test'], loc='upper right') 

  plt.show() 

#reload the data
import pickle

###########################################
X_train=pickle.load(open("/data/shangrui/c/X_train_512.pickle","rb"))##
y_train=pickle.load(open("/data/shangrui/c/y_train_512.pickle","rb"))##
X_test=pickle.load(open("/data/shangrui/c/X_test_512.pickle","rb"))##
y_test=pickle.load(open("/data/shangrui/c/y_test_512.pickle","rb"))##
###########################################


import tensorflow as tf
###########################
IMG_SHAPE = (512,512,3)##
###########################

# Create the base model from the pre-trained model InceptionV3
base_model = tf.keras.applications.InceptionV3(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable=False

#tf.keras.layers.GlobalAveragePooling2D layer to convert the features to a single 1280-element vector per image.
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

prediction_layer = tf.keras.layers.Dense(3,activation='softmax')

model= tf.keras.Sequential([
  base_model,
  global_average_layer,
  prediction_layer
])

model.compile(optimizer='Adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es=EarlyStopping(monitor='val_acc',mode='max',patience=8,restore_best_weights=True)

# fit model on the augmented dataset
###################################################################################################
training=model.fit(X_train,y_train,batch_size=8,validation_split=0.2,epochs=100,callbacks=[es])#
###################################################################################################

plot_lc(training,'loss')
plot_lc(training,'acc')

model.evaluate(X_test,y_test,batch_size=8)

model.summary()

############################################################
model_json=model.to_json()
with open("/home/shangrui/model1_c.json",'w') as json_file:
    json_file.write(model_json)

#--serialize weights to HDF5
model.save_weights('/home/shangrui/model1_c.h5')
print('Saved model')
############################################################











