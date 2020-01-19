import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time
import imutils

mixer.init()
sound = mixer.Sound('alarm.wav')
#辅助检测脸和眼睛的文件
face = cv2.CascadeClassifier('haar cascade files/haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files/haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files/haarcascade_righteye_2splits.xml')



lbl=['Close','Open']
#导入训练好的模型
model = load_model('models/cnncat2.h5')
path = os.getcwd()
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count=0
score=0
thicc=2
rpred=[99]
lpred=[99]

while(True):
    ret, frame = cap.read()
    frame = imutils.resize(frame,width=450)
    height,width = frame.shape[:2] 
    #将读取到的彩色图片转化为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #脸部特征点检测
    faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
    #左眼
    left_eye = leye.detectMultiScale(gray)
    #右眼
    right_eye =  reye.detectMultiScale(gray)
    #用来框住人脸的矩阵
    cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED )

    #随着人脸的移动，框也跟着移动
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )

    for (x,y,w,h) in right_eye:
        r_eye=frame[y:y+h,x:x+w]
        count=count+1
        r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye,(24,24))
        r_eye= r_eye/255
        r_eye=  r_eye.reshape(24,24,-1)
        r_eye = np.expand_dims(r_eye,axis=0)
        rpred = model.predict_classes(r_eye)
        if(rpred[0]==1):
            lbl='Open' 
        if(rpred[0]==0):
            lbl='Closed'
        break

    for (x,y,w,h) in left_eye:
        l_eye=frame[y:y+h,x:x+w]
        count=count+1
        l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  
        l_eye = cv2.resize(l_eye,(24,24))
        l_eye= l_eye/255
        l_eye=l_eye.reshape(24,24,-1)
        l_eye = np.expand_dims(l_eye,axis=0)
        lpred = model.predict_classes(l_eye)
        if(lpred[0]==1):
            lbl='Open'   
        if(lpred[0]==0):
            lbl='Closed'
        break
    #如果两只眼睛都是闭着
    if(rpred[0]==0 and lpred[0]==0):
        score=score+1
        cv2.putText(frame,"Closed",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    # if(rpred[0]==1 or lpred[0]==1):
    #这里考虑的情况就比较复杂
    #比如，单眼闭着，另一只眼睁着，看得见
    #两只眼同时睁着
    #如果是有眼部残疾的用户，一只眼睁闭状况可能归为同时睁眼或者闭眼
    else:
        score=score-1
        cv2.putText(frame,"Open",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)

    # 我们将会用score作为判断用户是否疲劳的依据
    # 这里有一个比较难确定的地方，都会遇到的问题就是，眨眼频率的阈值为多少
    # 才算是进入疲劳状态，这是计算机视觉感应的通病
    # 在代码里面我选择了 15 作为分界线
    # 当然，后续可以有更加好的修改，比如针对不同场景下，设置不同的阈值
    # 在高速公路上，或者进入交通拥挤的地段，阈值可以低一点
    # 在车流量、人流量较少的地段，阈值可以设置高一点
    # 于是我们可以利用这一点，设置一个系统的模式
    # 比如 拥挤模式、普通模式之类的
    # 可以结合 物体检测去设置阈值，一般是摄像头内，物体的分布个数可以作为评判的阈值
    # 后续如何做需要大家讨论
    if(score<0):
        score=0   
    cv2.putText(frame,'Score:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    if(score>15):
        # 说明此时人已经有点疲劳了，应该发出警报
        cv2.imwrite(os.path.join(path,'image.jpg'),frame)
        try:
            sound.play()
        #如果不是，我们就跳过
        except:  # isplaying = False
            pass
        if(thicc<16):
            thicc= thicc+2
        else:
            thicc=thicc-2
            if(thicc<2):
                thicc=2
        cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc) 
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
