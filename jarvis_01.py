import cv2 
import numpy as np
import lib.package.handtrackingmodule as htm
import time
import autopy

wCam,hCam = 640,480
frameR = 100 # Frame Reduction (range) (used in sensitivity)

smothening = 7
#location variable
plocx,plocy = 0,0
clocx,clocy = 0,0

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
cap.set(3,wCam)
cap.set(4,hCam)
pTime = 0

detector = htm.handDetector(maxHands=1)
ws,hs = autopy.screen.size()


while True:
    #getting the img
    success,img = cap.read()
    img = detector.findHands(img)
    lmlist,bbox = detector.findPosition(img)
    #get the tip of the index finger and middle finger
    if len(lmlist)!=0:
        x1,y1 = lmlist[8][1:]    #co-ordinates of index and middel finger
        x2,y2 = lmlist[12][1:]
        # print(x1,y1,x2,y2)
        
        #check which fingers are up
        fingers = detector.fingersUp()
        # print(fingers)
        cv2.rectangle(img,(frameR,frameR),(wCam-frameR,hCam-frameR),(255,0,255),2)

        #only index finger :moving mode
        if fingers[1] == 1 and fingers[2] ==0 :

            # convert coordinates
            x3 = np.interp(x1,(frameR,wCam-frameR),(0,ws))  #for sensitivity
            y3 = np.interp(y1,(frameR,hCam-frameR),(0,hs))
            
            #smothening
            clocx = plocx + (x3 - plocx) /smothening
            clocy = plocy + (y3 -plocy) /smothening
            

            #move our mouse
            autopy.mouse.move(ws-clocx,clocy) # changing the movement opposite (ws-x3,y3)
            cv2.circle(img,(x1,y1),15,(255,34,0),cv2.FILLED)
            plocy,plocx = clocy,clocx


        #both index and middle finger are up then it is clicking mode
        if fingers[1] == 1 and fingers[2] == 1 :
            
            #we will find the distance between the fingers
            length,img,linfo= detector.findDistance(8,12,img)#landmark id 8,12

            # if distance is small click the mouse
            if length < 40 :
                cv2.circle(img,(linfo[4],linfo[5]),15,(0,255,0),cv2.FILLED)
                autopy.mouse.click()

    #frame rate
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img,str(int(fps)),(10,50),cv2.FONT_HERSHEY_COMPLEX,3,(245,244,0),3)
    #to display 
    cv2.imshow('image',img)
    cv2.waitKey(1)
