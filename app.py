#importing the libraries
import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import keras
import numpy as np
import PIL

#initializing the counter variables
drone_counter=0
non_drone_counter=0

#model evaluation
model=load_model(r'D:\drone_project\VS\weight\densnet121.h5')
results=[]

#Checking accuracy for non drone elements
for i in range(82):
        path=r'D:\drone_project\VS\test\drone\{}.jpg'.format(i+1)
        img = keras.preprocessing.image.load_img(path, target_size=(640, 640))
            #converting it into array 
        x=keras.preprocessing.image.img_to_array(img)
        ## Scaling
        x=x/255

        x=np.expand_dims(x, axis=0)
            # calling the first model and passing the image for prediction
        preds1= model.predict(x)
        preds1=np.argmax(preds1, axis=1)

        if preds1[0]==0:
            print('This image contains a Drone')
            results.append(1)
            #number of right prediction for drone
            drone_counter=drone_counter+1
        else:
            print('This image does not have a drone')
            results.append(0)
            non_drone_counter=non_drone_counter+1

print('Drone Accuracy = ',(drone_counter/len(results)))




