##1.1.2 for loop to convert all
import numpy as np
import PIL
from PIL import Image
import os
import pandas as pd
#to iterate through directories join paths 

datadir="/data/shangrui/bsea/"


categories=['average','poor','wealthy']
training_data=[]


def create_training_data():
    for category in categories:
        path=os.path.join(datadir,category)
        class_num=categories.index(category)
        #converting class into numerical value
        for img in os.listdir(path):
            try:
                img=Image.open(os.path.join(path,img))
                imarray=np.array(img)
                
#read the image in grayscale via cv2.imread(fullpath,specify the way)
                training_data.append(imarray.shape)
                #append the image and correspond class into the list
            except Exception as e:
                pass
create_training_data()


training_data=pd.DataFrame(training_data)
#print(training_data)
print(training_data.describe())

   
    
 



