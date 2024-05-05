from flask import Flask,render_template,request,jsonify
import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import keras
import numpy as np
import PIL
import os
from werkzeug.utils import secure_filename

app=Flask(__name__)

model=load_model('D:\drone_project\VS\weight\densnet121.h5')

# Define the upload and process directories
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(PROCESSED_FOLDER):
    os.makedirs(PROCESSED_FOLDER)

@app.route('/')
def home_page():
    return render_template('home.html')


@app.route('/submit',methods=['POST'])
def image_selector():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        # Save the uploaded image to the 'uploads' directory
        filename = secure_filename(file.filename)
        save_dir=os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        # Process the image using your model
        result = process_image(save_dir)

        # Move the image to the 'processed' directory
        try:
            os.rename(os.path.join(app.config['UPLOAD_FOLDER'], filename), os.path.join(app.config['PROCESSED_FOLDER'], filename))
        except FileExistsError:
            pass            

        return jsonify({'result': result})


def process_image(filename):
    img = keras.preprocessing.image.load_img(filename, target_size=(640, 640))
            #converting it into array 
    x=keras.preprocessing.image.img_to_array(img)
                    ## Scaling
    x=x/255
    x=np.expand_dims(x, axis=0)
            # calling the  model and passing the image for prediction
    preds1= model.predict(x)
    preds1=np.argmax(preds1, axis=1)
    if preds1[0]==0:
        return ('This image contains a Drone')
    else:
        return ('This image does not have a drone')          




if __name__=='__main__':
    app.run(debug=True)
