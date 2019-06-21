# -*- coding: utf-8 -*-

#Face alignment with face image
import os
import cv2
from mtcnn.mtcnn import MTCNN
import dlib
from imutils.face_utils import FaceAligner
from imutils import face_utils

def listFiles(path):
    dirs = os.listdir(path)
    return dirs
    
def extract_feature(path):
    detector = MTCNN()
    predictor = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")
    fa = FaceAligner(predictor, desiredFaceWidth=256)
    files = listFiles(path)
    for f in files:
        image = cv2.imread("./"+path+"/"+f)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
        retcs = detector.detect_faces(image)
        if len(retcs) == 1 :#only one face
            box = retcs[0]['box']
            box_new = dlib.rectangle(box[0],box[1], box[0]+box[2], box[1]+box[3])
            faceAligned = fa.align(image, gray_image, box_new)    
            cv2.imshow(f, faceAligned)   
            cv2.imwrite(f,faceAligned)
            (x,y,w,h) = face_utils.rect_to_bb(box_new)
            cv2.rectangle(image, (x, y), (x+w, y+h), (0,0,255), 1)
        else:
            cv2.imshow("no face" + f, image)
        print(retcs)
        cv2.waitKey(0)
        cv2.destroyAllWindows()  

def main():
    path = "./data/Phuc"
    extract_feature(path)

if __name__=='__main__':
    main()