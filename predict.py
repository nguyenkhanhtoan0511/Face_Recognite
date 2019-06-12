# Xác nhận người trong ảnh có phải là người cần nhận diện hay không ?
import numpy as np
import os
# from sklearn.neighbors import KDTree
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

def load_features(src):
    print("[+] Load data....")
    data = []
    for file in os.listdir(src):
        data.append(np.load(os.path.join(src, file))[0])
        # print('file : ', file)
    print("[+] Load data finished")
    return data

def detect_face(path):
    detector = MTCNN()
    try:
        predictor = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")
    except:
        print('can not open file shape_predictor_68_face_landmarks.dat ')
    fa = FaceAligner(predictor, desiredFaceWidth=256)
    #read image
    img = cv2.imread(path)
    gray_image= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # if result is empty then we try to replace size image
    faces = detector.detect_faces(img)
    print(np.array(faces).shape)
    if len(faces) == 0  :
        return None, None, img
    box = faces[0]['box'] # only 1 face
    #convert dist to rtype: dlib.rectangle
    box_new = dlib.rectangle(box[0], box[1], box[0]+box[2], box[1]+box[3])
    #face alignment
    faceAligned = fa.align(img, gray_image,box_new)
    return faceAligned, box_new, img
    
def draw_rectange(box_new,img, text):
    (x,y,w,h) = face_utils.rect_to_bb(box_new)
    #draw on new image
    image_copy = img.copy()
    cv2.rectangle(image_copy, (x,y), (x+w, y+h), (0,0,255), 1)
    cv2.putText(image_copy, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
    return image_copy

def save_feature(save_path, feature):    
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print("[+]Save extracted feature to file: ", save_path)
        np.save(save_path, feature)

def predict_image(path):
    # load KDtree
    file = open('kdtree.pickle', 'rb')
    kdt = pickle.load(file)
    file.close()
    faceAligned, box_new, image_result = detect_face(path)
    if faceAligned is not None:
        # So khớp khuôn mặt 
        save_face = './results/face.jpg'
        cv2.imwrite(save_face,faceAligned) 
        ImageFile.LOAD_TRUNCATED_IMAGES=True
        print("[+] Setup model")
        base_model = VGG16(weights='imagenet', include_top=True)
        out = base_model.get_layer("fc2").output
        model_feature = Model(inputs=base_model.input, outputs=out)
        img_path = save_face 
        print("[+] Read image  : ", img_path)
        save_path = ''
        path_feature =  './features/predict_set' 
        if os.path.isfile(img_path) and img_path.find(".jpg") != -1:    
            import numpy as np        
            save_path_feature = img_path.replace("results", path_feature).replace(".jpg", ".npy")            
            print('path : ', save_path_feature)                       
            img = image.load_img(img_path, target_size=(224, 224))
            img_data = image.img_to_array(img)
            img_data = np.expand_dims(img_data, axis=0)
            img_data = preprocess_input(img_data)
            print("[+] Extract feature from image : ", img_path)
            feature = model_feature.predict(img_data)
            save_feature(save_path_feature, feature)
        # match between one indiced face and all faces in  data set
        data_predict = load_features(path_feature)
        X_pre = np.array(data_predict)
        distance, indices = kdt.query(X_pre, k=4, return_distance=True)
        name = "Unknown"
        print('distance: ',distance)
        if distance[0,0] <= 50 :
            print('face detected....')
            # part 3 : making new predictions
            from keras.models import load_model
            import numpy as np
            # identical to the previous one
            model = load_model('model.h5')
            test_image = np.expand_dims(faceAligned, axis=0)
            result = model.predict_classes(test_image)
            if result[0] == 0:
                name = "Phuc"
            elif  result[0] == 1:
                name = "Phuong"
            elif  result[0] == 2:
                name = "Toan"
        image_copy = draw_rectange(box_new, image_result, name)
        cv2.imshow('Face recognition', image_copy)
        cv2.imwrite('./results/kq.jpg',image_copy )
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print('no face')
        image_copy = image_result.copy()
        cv2.putText(image_copy, 'No face detection', (10, 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
        cv2.imshow('Face recognition', image_copy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    path = 'dataset/predict/t4.jpg' #  update there
    predict_image(path)

if __name__=='__main__':
    main()