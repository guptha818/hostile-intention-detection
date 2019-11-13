import cv2
from model_loader import FacialExpressionModel
import numpy as np

rgb = cv2.VideoCapture(0)#WEB CAM 1 
facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')#XM FILE THAT DETECTS FACES IN WINDOW AND RETURN REGION
font = cv2.FONT_HERSHEY_TRIPLEX#TEXT FORMAT

def __get_data__():
    _, img = rgb.read()
    img = cv2.flip(img,1)#IMAGE FLIP
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = facec.detectMultiScale(gray, 1.3, 5)
    
    return faces, img, gray

def start(cnn):
    ix = 0#VARIABLE INC
    while True:
        ix += 1
        
        faces, fr, gray_fr = __get_data__()
        for (x, y, w, h) in faces:#REGION OF FACE
            fc = gray_fr[y:y+h, x:x+w]
    
            roi = cv2.resize(fc, (48, 48))#IMAGE RESIZE
            pred = cnn.predict_emotion(roi[np.newaxis, :, :, np.newaxis])# CONVERTING ROI TO NUMPY ARRAY BECAUSE MODEL  ACCEPTS IN NUMPY ARRAY FORMAT.

            cv2.putText(fr, pred, (x, y), font, 1, (255, 255, 0), 2)#TEXT DISPLAY
            cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0),2)# RECTANGLE AREA THAT DISPLAY

        if cv2.waitKey(5) == 27:#WAITING FOR KEY STROKE FOR 5MS AND ASCII VALUE OF ESC IS 27
            rgb.release()
            break
        cv2.imshow('Filter', fr)
    cv2.destroyAllWindows()



model = FacialExpressionModel("face_model.json", "face_model.h5")#ALREADY TRAINED MODEL TKES 5-6 HRS TO TRAIN MODEL
start(model)#PASSING MODEL TO START FUNCTION TO DETECT.

#In keras, non-trainable parameters means the number of weights that are not updated during training with backpropagation.
