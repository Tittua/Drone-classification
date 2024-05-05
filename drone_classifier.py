#importing the libraries
import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import keras
import numpy as np
import PIL

model=load_model(r'D:\drone_project\VS\weight\densnet121.h5')
#Please enter the path of the image
path=''

img = keras.preprocessing.image.load_img(path, target_size=(640, 640))
#converting it into array 
x=keras.preprocessing.image.img_to_array(img)
        ## Scaling
x=x/255
x=np.expand_dims(x, axis=0)
# calling the  model and passing the image for prediction
preds1= model.predict(x)
preds1=np.argmax(preds1, axis=1)
if preds1[0]==0:
    print('This image contains a Drone')
else:
    print('This image does not have a drone')
        

