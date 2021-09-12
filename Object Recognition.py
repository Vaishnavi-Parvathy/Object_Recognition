#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import imutils
import cv2
import time



prototxt="MobileNetSSD_deploy.prototxt"
model="MobileNetSSD_deploy.caffemodel"
confThresh=0.2

CLASSES=["background","aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow","dinning_table","dog","horse","motor_bike","person","potte_plant","sheep","sofa","train","tvmonitor"]

COLORS=np.random.uniform(0,255,size=(len(CLASSESS),3))#different colour for multiple objects

print("Loading model..")
net=cv2.dnn.readNetFromCaffe(prototxt, model)
print("Model Loaded")
print("Starting Camera Feed...")
vs =  cv2.VideoCapture(0,cv2.CAP_DSHOW)
time.sleep(2.0)
while True:
    _,frame=vs.read()
    frame=imutils.resize(frame, width=600)
    (h,w)=frame.shape[:2]
    imResize=cv2.resize(frame,(300, 300))
    blob=cv2.dnn.blobFromImage(imResize,0.007843,(300,300),127.5)
    net.setInput(blob)
    detections=net.forward()#gives boundry class id
    detShape=net.forward()
    detShape=detections.shape[2]
    for i in np.arange(0,detShape):
        confidence=detections[0,0,i,2] #number of percent matching with obj
        if confidence> confThresh:
            idx=int(detections[0,0,i,1])
            box=detections[0,0,i,3:7]*np.array([w,h,w,h])
            (startX,startY,endX,endY)=box.astype("int")
            label="{}: {:.2f}%".format(CLASSES[idx],confidence*100)
            cv2.rectangle(frame,(startX,startY),(endX,endY),COLORS[idx],2)
            if startY-15>15:#(frame big na inside the bounding box text is written else outside the bounding box)
                y=startY-15
            else:
                y=startY+15
            cv2.putText(frame,label,(startX,y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx],2)
    cv2.imshow("Frame", frame)
    key=cv2.waitKey(1)
    if key==27:
        break
cam.release()
cv2.destroyAllWindows()
                        
                        

