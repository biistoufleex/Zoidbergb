from flask import Flask, flash, redirect, request, render_template
import tensorflow as tf
import numpy as np
import os

app = Flask(__name__)

# set upload folder
UPLOAD_FOLDER = 'static/images'
CHECKPOINT_DIR = 'saved_model/'
MODEL_SAVED = CHECKPOINT_DIR + 'latest_model.h5'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# # set variable for model
class_names=['NORMAL', 'PNEUMONIA']

# load model
new_model = tf.keras.models.load_model(MODEL_SAVED)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'image' not in request.files:
            flash('No file part')
            return redirect(request.url)
        image = request.files['image']
        # if user does not select file, browser also
        # submit an empty part without filename
        if image.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if image:
            filename = image.filename
            basedir = os.path.abspath(os.path.dirname(__file__))
            image.save(os.path.join(basedir,app.config["UPLOAD_FOLDER"],filename))
            
            result = get_prediction(os.path.join(basedir,app.config["UPLOAD_FOLDER"],filename))
            return render_template('index.html', image_name=filename, prediction=result)

    return render_template('index.html')

def get_prediction(image_name):
    img = tf.keras.utils.load_img(
        image_name, target_size=(128, 128), color_mode="grayscale"
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = new_model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    result = "This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score))
    print(result)

    return result

if __name__ == '__main__':
    app.run(debug=True)