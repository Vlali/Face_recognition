import cv2
#you cant record audio with opencv,if you want you can use ffmpeg
cap=cv2.VideoCapture(0)

#Imports a xml module,which will recognize your face
face_cascade=cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_alt2.xml')
while True:
    ret,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    #face recognizing
    #we must use gray because this xml detect face in gray,after the cordinates,we can draw a rectangle around your face
    faces=face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)
    for(x,y,w,h) in faces:
        print(x,y,w,h)
        roi_gray=gray[y:y+h,x:x+w]   #cordinates
        roi_color=frame[y:y+h,x:x+w]
        
        #creates an image of your face
        img_item='my-image.png'
        cv2.imwrite(img_item,roi_gray) 
        
        #draw a rectangle around your face
        color=(255,0,0)  #BGR
        stroke=2
        end_cord_x=x+w
        end_cord_y=y+h
        cv2.rectangle(frame,(x,y),(end_cord_x,end_cord_y),color,stroke)
        
    #display the resulting frame              
    cv2.imshow('frame',frame)

    if cv2.waitKey(50) & 0xFF == ord('q'):  # to quit a webcam
        break


cap.release()
cv2.destroyAllWindows()