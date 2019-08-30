import numpy as np
import PIL
from PIL import Image
import os

#to iterate through directories join paths 
datadir="/data/shangrui/a/"


categories=['average','poor','wealthy']
training_data=[]
size=512,512

def create_training_data():
    for category in categories:
        path=os.path.join(datadir,category)
        class_num=categories.index(category)
        #converting class into numerical value
        for img in os.listdir(path):
            try:
                img=Image.open(os.path.join(path,img))
                img=img.resize(size,Image.ANTIALIAS)
                imarray=np.array(img)
                #read the image in grayscale via cv2.imread(fullpath,specify the way)
                training_data.append([imarray,class_num])
                #append the image and correspond class into the list
            except Exception as e:
                pass
create_training_data()
print(len(training_data))


#shuffle the data since the image is read follow the order
import random

random.shuffle(training_data)

#checking if traing_data is shuffled 
for sample in training_data[:10]:
    print(sample[1])


#split into features and coverting X type into np.array for put into CNN
X=[]
y=[]

#standardize X
for features,label in training_data:
    X.append(features)
    y.append(label)
#list is not acceptale as a input into CNN (y can stay as a list while X cannot and X has to be a numpy array)

X=np.array(X)

#saving the data 
import pickle

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pickle_out=open("/data/shangrui/a/X_train_512.pickle","wb")
#open the file for wrting "X.pickle"is the file name
pickle.dump(X_train,pickle_out)
#write the object X to the file "X.pickle"
pickle_out.close()
#close the object

pickle_out=open("/data/shangrui/a/y_train_512.pickle",'wb')
pickle.dump(y_train,pickle_out)
pickle_out.close()

pickle_out=open("/data/shangrui/a/X_test_512.pickle","wb")
pickle.dump(X_test,pickle_out)
pickle_out.close()

pickle_out=open("/data/shangrui/a/y_test_512.pickle","wb")
pickle.dump(y_test,pickle_out)
pickle_out.close()


def horizontal_flip(image_array):
    #horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !
    return image_array[:, ::-1]

X_train_aug=[]
y_train_aug=[]
for i in range(len(X_train)):
  img_arrary_aug=horizontal_flip(X_train[i])
  #v=np.flipud(X_train[i])
  X_train_aug.append(X_train[i])
  y_train_aug.append(y_train[i])
  X_train_aug.append(img_arrary_aug)
  y_train_aug.append(y_train[i])
  #X_train_aug.append(v)
  #y_train_aug.append(y_train[i])

training_aug=[]
for i in range(len(X_train_aug)):
    training_data.append([X_train_aug[i],y_train_aug[i]])
random.shuffle(training_aug)

for features,label in training_aug:
    X_train_aug.append(features)
    y_train_aug.append(label)

X_train_aug=np.array(X_train_aug)
y_train_aug=np.array(y_train_aug)

pickle_out=open("/data/shangrui/a/X_train_aug512.pickle","wb")
pickle.dump(X_train_aug,pickle_out)
pickle_out.close()

pickle_out=open("/data/shangrui/a/y_train_aug512.pickle",'wb')
pickle.dump(y_train_aug,pickle_out)
pickle_out.close()

pickle_out=open("/data/shangrui/a/X_test_aug512.pickle","wb")
pickle.dump(X_test,pickle_out)
pickle_out.close()

pickle_out=open("/data/shangrui/a/y_test_aug512.pickle","wb")
pickle.dump(y_test,pickle_out)
pickle_out.close()

















