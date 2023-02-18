import os
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request, render_template
from FlaskBackend.modules.database import collection as db
from datetime import datetime

app = Flask(__name__)

tflite_model_file_seedlings = 'seedlings.tflite'
with open(tflite_model_file_seedlings, 'rb') as fid:
    tflite_model_seedlings = fid.read()

tflite_model_file_flowers = 'my_flowers.tflite'
with open(tflite_model_file_flowers, 'rb') as fid:
    tflite_model_flowers = fid.read()

tflite_model_file_leaves = 'leaf.tflite'
with open(tflite_model_file_leaves, 'rb') as fid:
    tflite_model_leaves = fid.read()

tflite_model_file_weed = 'my_weed.tflite'
with open(tflite_model_file_weed, 'rb') as fid:
    tflite_model_weed = fid.read()

tflite_model_file_paddy = 'paddy.tflite'
with open(tflite_model_file_paddy, 'rb') as fid:
    tflite_model_paddy = fid.read()

tflite_model_file_crop = 'crop.tflite'
with open(tflite_model_file_crop, 'rb') as fid:
    tflite_model_crop = fid.read()

target_img = os.path.join(os.getcwd(), 'static/images')


# Function to load and prepare the image in right shape
def read_image_seed(filename, size):
    img = load_img(filename, target_size=size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


# Allow files with extension png, jpg and jpeg
ALLOWED_EXT = {'jpg', 'jpeg', 'png'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT


@app.route('/seedlings', methods=['POST'])
def predict_seedlings():
    global seed
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):  # Checking file format
            filename = file.filename
            file_path = os.path.join('static/images/seedlings', filename)
            file.save(file_path)
            img_size = (96, 96)
            img = read_image_seed(file_path, img_size)  # preprocessing method

            # Load the TFLite model and allocate tensors
            interpreter = tf.lite.Interpreter(model_content=tflite_model_seedlings)
            interpreter.allocate_tensors()

            # Get input and output tensors
            input_details = interpreter.get_input_details()[0]['index']
            output_details = interpreter.get_output_details()[0]['index']
            class_prediction = []

            interpreter.set_tensor(input_details, img)
            interpreter.invoke()
            class_prediction.append(interpreter.get_tensor(output_details))

            classes_x = np.argmax(class_prediction)

            if classes_x == 0:
                seed = "Scentless Mayweed"
            elif classes_x == 1:
                seed = "Common wheat"
            elif classes_x == 2:
                seed = "Charlock"
            elif classes_x == 3:
                seed = "Black grass"
            elif classes_x == 4:
                seed = "Sugar beet"
            elif classes_x == 5:
                seed = "Loose Silky-bent"
            elif classes_x == 6:
                seed = "Maize"
            elif classes_x == 7:
                seed = "Cleavers"
            elif classes_x == 8:
                seed = "Common Chickweed"
            elif classes_x == 9:
                seed = "Fat Hen"
            elif classes_x == 10:
                seed = "Small-flowered Cranesbill"
            elif classes_x == 11:
                seed = "Shepherd’s Purse"

            db.addNewImage(
                file.filename,
                seed,
                datetime.now(),
                target_img + '/seedlings' + file.filename)

            return jsonify(prediction=seed, user_image=file_path)
        else:
            return "Unable to read the file. Please check file extension"


@app.route('/flowers', methods=['POST'])
def predict_flowers():
    global flower
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):  # Checking file format
            filename = file.filename
            file_path = os.path.join('static/images/flowers', filename)
            file.save(file_path)
            img_size = (224, 224)
            img = read_image_seed(file_path, img_size)  # preprocessing method

            # Load the TFLite model and allocate tensors
            interpreter = tf.lite.Interpreter(model_content=tflite_model_flowers)
            interpreter.allocate_tensors()

            # Get input and output tensors
            input_details = interpreter.get_input_details()[0]['index']
            output_details = interpreter.get_output_details()[0]['index']
            class_prediction = []

            interpreter.set_tensor(input_details, img)
            interpreter.invoke()
            class_prediction.append(interpreter.get_tensor(output_details))

            classes_x = np.argmax(class_prediction)

            if classes_x == 0:
                flower = "Dandelion"
            elif classes_x == 1:
                flower = "Daisy"
            elif classes_x == 2:
                flower = "Tulip"
            elif classes_x == 3:
                flower = "Sunflower"
            elif classes_x == 4:
                flower = "Rose"

            db.addNewImage(
                file.filename,
                flower,
                datetime.now(),
                target_img + '/flowers' + file.filename)

            return jsonify(prediction=flower, user_image=file_path)
        else:
            return "Unable to read the file. Please check file extension"


# needs to modified (image size)
@app.route('/leaves', methods=['POST'])
def predict_leaves():
    global leaf
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):  # Checking file format
            filename = file.filename
            file_path = os.path.join('static/images/leaves', filename)
            file.save(file_path)
            img_size = (224, 224)
            img = read_image_seed(file_path, img_size)  # preprocessing method

            # Load the TFLite model and allocate tensors
            interpreter = tf.lite.Interpreter(model_content=tflite_model_leaves)
            interpreter.allocate_tensors()

            # Get input and output tensors
            input_details = interpreter.get_input_details()[0]['index']
            output_details = interpreter.get_output_details()[0]['index']
            class_prediction = []

            interpreter.set_tensor(input_details, img)
            interpreter.invoke()
            class_prediction.append(interpreter.get_tensor(output_details))

            classes_x = np.argmax(class_prediction)

            if classes_x == 0:
                leaf = 'Acer Capillipes'
            elif classes_x == 1:
                leaf = 'Acer Circinatum'
            elif classes_x == 2:
                leaf = 'Acer Mono'
            elif classes_x == 3:
                leaf = 'Acer Opalus'
            elif classes_x == 4:
                leaf = 'Acer Palmatum'
            elif classes_x == 5:
                leaf = 'Acer Pictum'
            elif classes_x == 6:
                leaf = 'Acer Platanoids'
            elif classes_x == 7:
                leaf = 'Acer Rubrum'
            elif classes_x == 8:
                leaf = 'Acer Rufinerve'
            elif classes_x == 9:
                leaf = 'Acer Saccharinum'
            elif classes_x == 10:
                leaf = 'Alnus Cordata'
            elif classes_x == 11:
                leaf = 'Alnus Maximowiczii'
            elif classes_x == 12:
                leaf = 'Alnus Rubra'
            elif classes_x == 13:
                leaf = 'Alnus Sieboldiana'
            elif classes_x == 14:
                leaf = 'Alnus Viridis'
            elif classes_x == 15:
                leaf = 'Arundinaria Simonii'
            elif classes_x == 16:
                leaf = 'Betula Austrosinensis'
            elif classes_x == 17:
                leaf = 'Betula Pendula'
            elif classes_x == 18:
                leaf = 'Callicarpa Bodinieri'
            elif classes_x == 19:
                leaf = 'Castanea Sativa'
            elif classes_x == 20:
                leaf = 'Celtis Koraiensis'
            elif classes_x == 21:
                leaf = 'Cercis Siliquastrum'
            elif classes_x == 22:
                leaf = 'Cornus Chinensis'
            elif classes_x == 23:
                leaf = 'Cornus Controversa'
            elif classes_x == 24:
                leaf = 'Cornus Macrophylla'
            elif classes_x == 25:
                leaf = 'Cotinus Coggygria'
            elif classes_x == 26:
                leaf = 'Crataegus Monogyna'
            elif classes_x == 27:
                leaf = 'Cytisus Battandieri'
            elif classes_x == 28:
                leaf = 'Eucalyptus Glaucescens'
            elif classes_x == 29:
                leaf = 'Eucalyptus Neglecta'
            elif classes_x == 30:
                leaf = 'Eucalyptus Urnigera'
            elif classes_x == 31:
                leaf = 'Fagus Sylvatica'
            elif classes_x == 32:
                leaf = 'Ginkgo Biloba'
            elif classes_x == 33:
                leaf = 'Ilex Aquifolium'
            elif classes_x == 34:
                leaf = 'Ilex Cornuta'
            elif classes_x == 35:
                leaf = 'Liquidambar Styraciflua'
            elif classes_x == 36:
                leaf = 'Liriodendron Tulipifera'
            elif classes_x == 37:
                leaf = 'Lithocarpus Cleistocarpus'
            elif classes_x == 38:
                leaf = 'Lithocarpus Edulis'
            elif classes_x == 39:
                leaf = 'Magnolia Heptapeta'
            elif classes_x == 40:
                leaf = 'Magnolia Salicifolia'
            elif classes_x == 41:
                leaf = 'MorusNigra'
            elif classes_x == 42:
                leaf = 'Olea Europaea'
            elif classes_x == 43:
                leaf = 'Phildelphus'
            elif classes_x == 44:
                leaf = 'Populus Adenopoda'
            elif classes_x == 45:
                leaf = 'Populus Grandidentata'
            elif classes_x == 46:
                leaf = 'Populus Nigra'
            elif classes_x == 47:
                leaf = 'Prunus Avium'
            elif classes_x == 48:
                leaf = 'Prunus x Shmittii'
            elif classes_x == 49:
                leaf = 'Pterocarya Stenoptera'
            elif classes_x == 50:
                leaf = 'Quercus Afares'
            elif classes_x == 51:
                leaf = 'Quercus Agrifolia'
            elif classes_x == 52:
                leaf = 'Quercus Alnifolia'
            elif classes_x == 53:
                leaf = 'Quercus Brantii'
            elif classes_x == 54:
                leaf = 'Quercus Canariensis'
            elif classes_x == 55:
                leaf = 'Quercus Castaneifolia'
            elif classes_x == 56:
                leaf = 'Quercus Cerris'
            elif classes_x == 57:
                leaf = 'Quercus Chrysolepis'
            elif classes_x == 58:
                leaf = 'Quercus Coccifera'
            elif classes_x == 59:
                leaf = 'Quercus Coccinea'
            elif classes_x == 60:
                leaf = 'Quercus Crassifolia'
            elif classes_x == 61:
                leaf = 'Quercus Crassipes'
            elif classes_x == 62:
                leaf = 'Quercus Dolicholepis'
            elif classes_x == 63:
                leaf = 'Quercus Ellipsoidalis'
            elif classes_x == 64:
                leaf = 'Quercus Greggii'
            elif classes_x == 65:
                leaf = 'Quercus Hartwissiana'
            elif classes_x == 66:
                leaf = 'Quercus Ilex'
            elif classes_x == 67:
                leaf = 'Quercus Imbricaria'
            elif classes_x == 68:
                leaf = 'Quercus Infectoria_sub'
            elif classes_x == 69:
                leaf = 'Quercus Kewensis'
            elif classes_x == 70:
                leaf = 'Quercus Nigra'
            elif classes_x == 71:
                leaf = 'Quercus Palustris'
            elif classes_x == 72:
                leaf = 'Quercus Phellos'
            elif classes_x == 73:
                leaf = 'Quercus Phillyraeoides'
            elif classes_x == 74:
                leaf = 'Quercus Pontica'
            elif classes_x == 75:
                leaf = 'Quercus Pubescens'
            elif classes_x == 76:
                leaf = 'Quercus Pyrenaica'
            elif classes_x == 77:
                leaf = 'Quercus Rhysophylla'
            elif classes_x == 78:
                leaf = 'Quercus Rubra'
            elif classes_x == 79:
                leaf = 'Quercus Semecarpifolia'
            elif classes_x == 80:
                leaf = 'Quercus Shumardii'
            elif classes_x == 81:
                leaf = 'Quercus Suber'
            elif classes_x == 82:
                leaf = 'Quercus Texana'
            elif classes_x == 83:
                leaf = 'Quercus Trojana'
            elif classes_x == 84:
                leaf = 'Quercus Variabilis'
            elif classes_x == 85:
                leaf = 'Quercus Vulcanica'
            elif classes_x == 86:
                leaf = 'Quercus x Hispanica'
            elif classes_x == 87:
                leaf = 'Quercus x Turneri'
            elif classes_x == 88:
                leaf = 'Rhododendron x Russellianum'
            elif classes_x == 89:
                leaf = 'Salix Fragilis'
            elif classes_x == 90:
                leaf = 'Salix Intergra'
            elif classes_x == 91:
                leaf = 'Sorbus Aria'
            elif classes_x == 92:
                leaf = 'Tilia Oliveri'
            elif classes_x == 93:
                leaf = 'Tilia Platyphyllos'
            elif classes_x == 94:
                leaf = 'Tilia Tomentosa'
            elif classes_x == 95:
                leaf = 'Ulmus Bergmanniaa'
            elif classes_x == 96:
                leaf = 'Viburnum Tinus'
            elif classes_x == 97:
                leaf = 'Viburnum x Rhytidophylloides'
            elif classes_x == 98:
                leaf = 'Zelkova Serrata'
            db.addNewImage(
                file.filename,
                leaf,
                datetime.now(),
                target_img + '/leaves' + file.filename)

            return jsonify(prediction=leaf, user_image=file_path)
        else:
            return "Unable to read the file. Please check file extension"


@app.route('/paddy', methods=['POST'])
def predict_paddy():
    global disease
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):  # Checking file format
            filename = file.filename
            file_path = os.path.join('static/images/paddy', filename)
            file.save(file_path)
            img_size = (256, 256)
            img = read_image_seed(file_path, img_size)  # preprocessing method

            # Load the TFLite model and allocate tensors
            interpreter = tf.lite.Interpreter(model_content=tflite_model_paddy)
            interpreter.allocate_tensors()

            # Get input and output tensors
            input_details = interpreter.get_input_details()[0]['index']
            output_details = interpreter.get_output_details()[0]['index']
            class_prediction = []

            interpreter.set_tensor(input_details, img)
            interpreter.invoke()
            class_prediction.append(interpreter.get_tensor(output_details))

            classes_x = np.argmax(class_prediction)

            if classes_x == 0:
                disease = "Bacterial Leaf Blight"
            elif classes_x == 1:
                disease = "Bacterial Leaf Streak"
            elif classes_x == 2:
                disease = "Bacterial Panicle Blight"
            elif classes_x == 3:
                disease = "Blast"
            elif classes_x == 4:
                disease = "Brown Spot"
            elif classes_x == 5:
                disease = "Dead Heart"
            elif classes_x == 6:
                disease = "Downy Mildew"
            elif classes_x == 7:
                disease = "Hispa"
            elif classes_x == 8:
                disease = "Normal"
            elif classes_x == 9:
                disease = "Tungro"

            db.addNewImage(
                file.filename,
                disease,
                datetime.now(),
                target_img + '/paddy' + file.filename)

            return jsonify(prediction=disease, user_image=file_path)
        else:
            return "Unable to read the file. Please check file extension"


@app.route('/weeds', methods=['POST'])
def predict_weeds():
    global weed
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):  # Checking file format
            filename = file.filename
            file_path = os.path.join('static/images/weed', filename)
            file.save(file_path)
            img_size = (224, 224)
            img = read_image_seed(file_path, img_size)  # preprocessing method

            # Load the TFLite model and allocate tensors
            interpreter = tf.lite.Interpreter(model_content=tflite_model_weed)
            interpreter.allocate_tensors()

            # Get input and output tensors
            input_details = interpreter.get_input_details()[0]['index']
            output_details = interpreter.get_output_details()[0]['index']
            class_prediction = []

            interpreter.set_tensor(input_details, img)
            interpreter.invoke()
            class_prediction.append(interpreter.get_tensor(output_details))

            classes_x = np.argmax(class_prediction)

            if classes_x == 0:
                weed = "Nutsedge"
            elif classes_x == 1:
                weed = "Sicklepod"
            elif classes_x == 2:
                weed = "Morningglory"
            elif classes_x == 3:
                weed = "Ragweed"
            elif classes_x == 4:
                weed = "Palmer Amaranth"
            elif classes_x == 5:
                weed = "Waterhemp"
            elif classes_x == 6:
                weed = "Crabgrass"
            elif classes_x == 7:
                weed = "Swinecress"
            elif classes_x == 8:
                weed = "Prickly Sida"
            elif classes_x == 9:
                weed = "Carpet weeds"
            elif classes_x == 10:
                weed = "Spotted Spurge"
            elif classes_x == 11:
                weed = "SpurredAnoda"
            elif classes_x == 12:
                weed = "Eclipta"
            elif classes_x == 13:
                weed = "Goosegrass"
            elif classes_x == 14:
                weed = "Purslane"

            db.addNewImage(
                file.filename,
                weed,
                datetime.now(),
                target_img + '/weed' + file.filename)

            return jsonify(prediction=weed, user_image=file_path)
        else:
            return "Unable to read the file. Please check file extension"


@app.route('/crop', methods=['POST'])
def predict_crop():
    global classes_x, crop
    if request.method == 'POST':
        data = request.get_json()
        N = data['Nitrogen']
        P = data['Phosphorus']
        K = data['Potassium']
        temp = data['Temp']
        humidity = data['Humidity']
        Ph = data['Ph']
        rainFall = data['rainFall']

        query = np.array([N, P, K, temp, humidity, Ph, rainFall], dtype=np.float32)
        input_data = query.reshape(1, 7, 1)

        interpreter = tf.lite.Interpreter(model_content=tflite_model_crop)
        interpreter.allocate_tensors()

        # Get input and output tensors.
        input_details = interpreter.get_input_details()[0]['index']
        output_details = interpreter.get_output_details()[0]['index']

        interpreter.set_tensor(input_details, input_data)

        # Get input and output tensors

        class_prediction = []
        # interpreter.set_tensor(input_details, query)
        interpreter.invoke()
        class_prediction.append(interpreter.get_tensor(output_details))

        classes_x = np.argmax(class_prediction)

    if classes_x == 0:
        crop = 'Apple'
    elif classes_x == 1:
        crop = 'Banana'
    elif classes_x == 2:
        crop = 'Blackgram'
    elif classes_x == 3:
        crop = 'Chickpea'
    elif classes_x == 4:
        crop = 'Coconut'
    elif classes_x == 5:
        crop = 'Coffee'
    elif classes_x == 6:
        crop = 'Cotton'
    elif classes_x == 7:
        crop = 'Grapes'
    elif classes_x == 8:
        crop = 'jute'
    elif classes_x == 9:
        crop = 'Kidneybeans'
    elif classes_x == 10:
        crop = 'Lentil'
    elif classes_x == 11:
        crop = 'Maize'
    elif classes_x == 12:
        crop = 'Mango'
    elif classes_x == 13:
        crop = 'Mothbeans'
    elif classes_x == 14:
        crop = 'Mungbean'
    elif classes_x == 15:
        crop = 'Muskmelon'
    elif classes_x == 16:
        crop = 'Orange'
    elif classes_x == 17:
        crop = 'Papaya'
    elif classes_x == 18:
        crop = 'Pigeonpeas'
    elif classes_x == 19:
        crop = 'Pomegranate'
    elif classes_x == 20:
        crop = 'Rice'
    elif classes_x == 21:
        crop = 'Watermelon'

    db.cropRecommendation(
        crop,
        datetime.now())

    return jsonify(prediction=crop)


@app.route('/report', methods=['POST'])
def report():
    pollutionImage = request.json['image']
    location = request.json['location']
    db.pollutionReport(
        pollutionImage.filename,
        datetime.now(),
        location)
    return jsonify(location=location)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, port=9874)
