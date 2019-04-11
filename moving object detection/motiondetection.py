#import required packages
import cv2
import numpy as np
#create object for vide capture 0 is for your laptop's webcam
cap=cv2.VideoCapture(0)
a='path to store video example C:\\a.mp4'
codec=cv2.VideoWriter_fourcc('X','V','I','D')
vo=cv2.VideoWriter(a,codec,30,(640,480))
first=None
while True:
    #read frames,f denote frame and r has boolean value whether camera connected or not
    r,f=cap.read()
    #split image into different channel i.e,bgr
    b=f[:,:,0]
    g=f[:,:,1]
    r=f[:,:,2]
    #converting color into grayscale image
    gray=np.matrix(b/3+g/3+r/3,dtype='uint8')
    #applying filter for bluring image to reduce noice in image
    k1 = np.ones((3,3),np.uint8)/9
    gray=cv2.filter2D(gray,-1,k1)
    #save first image into first variable
    if first is None:
        first=gray
    #find absolute difference between each frame anf first frame
    gray1=cv2.absdiff(first,gray)
    #apply threshold if pixel value less than 20 it will become 0 else it will become 255
    th=cv2.threshold(gray1,20,255,cv2.THRESH_BINARY)[1]
    #this will increase the width of white border
    kernel = np.ones((5,5),np.uint8)
    th1=cv2.filter2D(th,-1,kernel)
    #this 3X3kernel is for canny edge detection.we find edges using matrix convolution
    k=np.matrix([[-1,-1,-1],
                 [-1,8,-1],
                 [-1,-1,-1]])
    cann1=cv2.filter2D(th1,-1,k)
    #we need to find pixel position having value 255 so we use below function
    img,c,h=cv2.findContours(cann1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    font = cv2.FONT_HERSHEY_SIMPLEX
    z=0
    for cnts in c:
        x=min(cnts[:,:,0])
        y=min(cnts[:,:,1])
        w=max(cnts[:,:,0])
        h=max(cnts[:,:,1])
        #(w-x)*(h-y) will give area of rectangle if it too less then we ignore it
        if (w-x)*(h-y)<10000:
            continue
        cv2.rectangle(f,(x,y),(w,h),(0,250,0),1)
        #we put some text on image if motion detected
        cv2.putText(f,'movement detected',(10,50), font, 1,(0,0,255),1,cv2.LINE_AA)
        cv2.imwrite('C:\\Users\\Akshay\\Downloads\\imageprocessing\\a.jpg',f)
        z=1
    if z==0:
        cv2.putText(f,'No movement detected',(10,50), font, 1,(255,255,255),1,cv2.LINE_AA)
    z=0
    cv2.imshow('original',f)
    cv2.imshow('filter',gray)
    cv2.imshow('dilate',th1)
    cv2.imshow('canny',cann1)
    vo.write(f)
    key=cv2.waitKey(1)
    if key & 0xff==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
