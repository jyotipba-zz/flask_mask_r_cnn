# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect, flash
from werkzeug.utils import secure_filename
from form import ImageForm

# utilities
import base64
import numpy as np
from PIL import Image
from io import BytesIO

# mask r cnn
from mrcnn.model import MaskRCNN
from keras.backend import clear_session

# helper function and class
from detection import PredictionConfig, make_prediction
from visulization import display_instances

cfg = PredictionConfig()

# load model weight

model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
model.load_weights('../Mask_RCNN/mask_rcnn_crossings_confidence_85_0043.h5', by_name=True)
model.keras_model._make_predict_function()  # This is very important

# Declare a flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'This should really be secret'

@app.route('/', methods=['GET', 'POST'])
def index():
    #
    form = ImageForm()
    if form.validate_on_submit():
        image = (form.image.data.read())
        pil_img = (Image.open(BytesIO(image))) # open in pil image

        result = make_prediction(pil_img, model)

        #intel = str(base64.b64encode(buffered).decode('utf-8'))
        #print(img_str)
        #print(result)
        detected_img = display_instances(pil_img, result['rois'], result['masks'],result['class_ids'],['bg', 'c', 'n'], result['scores'],figsize=(8,8))
        buffered = BytesIO()
        d_img = Image.fromarray(detected_img, 'RGB')
        d_img.save(buffered, format="png")
        img_str = (base64.b64encode(buffered.getvalue())).decode('utf-8')
        return render_template('index.html', form = form, img = img_str )



    

    return render_template('index.html', form = form )


if __name__ == '__main__':
    app.run('0.0.0.0', 5000, debug=True)
