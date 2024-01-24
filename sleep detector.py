import numpy as np
from keras.models import load_model
import cv2

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print('cannot open camera')
    exit()

face_detector = cv2.CascadeClassifier('haar-cascade-files/haarcascade_frontalface_alt.xml')
left_eye_detector = cv2.CascadeClassifier('haar-cascade-files/haarcascade_lefteye_2splits.xml')
right_eye_detector = cv2.CascadeClassifier('haar-cascade-files/haarcascade_righteye_2splits.xml')
model = load_model('models/my_model.h5')
labels = ['Closed', 'Open']
lpred=[20]
rpred=[20]
score=0

font = cv2.FONT_HERSHEY_COMPLEX_SMALL

while True:
    ret, frame = cap.read()

    if not ret:
        print('Cant receive frame')
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    left_eye = left_eye_detector.detectMultiScale(gray, 1.3, 5)
    right_eye = right_eye_detector.detectMultiScale(gray, 1.3, 5)
    face = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in face:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (100,100,100), 2)

    for (x,y,w,h) in left_eye:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)
        l_eye = frame[y:y+h,x:x+w]
        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
        l_eye = cv2.resize(l_eye,(24,24))
        l_eye = l_eye.reshape(24,24,1)
        l_eye = np.expand_dims(l_eye, axis=0)
        lpred = np.argmax(model.predict(l_eye),axis=1)
        break

    for (x,y,w,h) in right_eye:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
        r_eye = frame[y:y+h,x:x+w]
        r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye,(24,24))
        r_eye = r_eye.reshape(24,24,1)
        r_eye = np.expand_dims(r_eye, axis=0)
        rpred = np.argmax(model.predict(r_eye),axis=1)
        break

    cv2.putText(frame, f'Left: {labels[lpred[0]]}', (50,50), font, 2, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(frame, f'Right: {labels[rpred[0]]}', (50,100), font, 2, (255,255,255), 1, cv2.LINE_AA)
    
    if lpred[0] == 0 and rpred[0] == 0:
        score+=1
    elif score<=0:
        score=0
    else:
        score-=1
    cv2.putText(frame, f'Score: {score}', (50,150), font, 2, (255,255,255), 1, cv2.LINE_AA)

    if score>15:
        cv2.putText(frame, 'You fell asleep', (50,200), font, 2, (1,1,1), 2, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1)==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()    
        
        
    
