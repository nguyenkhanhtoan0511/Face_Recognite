# Xác nhận người trong ảnh có phải là người cần nhận diện hay không ?
import numpy as np
import os
from sklearn.neighbors import KDTree
 # Gọi lớp extrac_features.py để trích xuất đặc trưng từ face
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.applications.vgg16 import preprocess_input
from PIL import ImageFile
import cv2
from mtcnn.mtcnn import MTCNN
import dlib
from imutils.face_utils import FaceAligner
from imutils import face_utils
import pickle
import warnings
from flask import Flask, render_template, request
import base64
import predict
from PIL import Image

app = Flask(__name__)
UPLOAD_FOLDER = os.path.basename('predict')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def start_page():
    print("Start")
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['image']
    # Read image
    image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    filename = 'dataset/predict/'+file.filename 
    #image.save(filename)
    cv2.imwrite(filename,image)
    path='dataset/predict/'+file.filename
    image_result,name=predict.predict_image(path)
    image_content = cv2.imencode('.jpg', image_result)[1].tostring()
    encoded_image = base64.encodestring(image_content)
    to_send = 'data:image/jpg;base64, ' + str(encoded_image, 'utf-8')
    return render_template('index.html', faceDetected=1, num_faces=1, image_to_show=to_send, init=True,name=name)

if __name__ == "__main__":
    # Only for debugging while developing
    app.run(host='0.0.0.0', debug=True, port=9999)
       
