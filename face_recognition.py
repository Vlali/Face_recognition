import cv2
import pickle
#you cant record audio with opencv,if you want you can use ffmpeg


#Imports a xml module,which will recognize your face
face_cascade=cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_alt2.xml')
recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainner.yml')

labels={'Lali':1,'Anyu':0}
with open('labels.pickle','rb') as f:
    og_labels=pickle.load(f)
    labels={value:key for key,value in og_labels.items()}

cap=cv2.VideoCapture(0)

while True:
    ret,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    #face recognizing
    #we must use gray because this xml detect face in gray,after the cordinates,we can draw a rectangle around your face
    faces=face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)
    for(x,y,w,h) in faces:
        #print(x,y,w,h)
       
        roi_gray=gray[y:y+h,x:x+w]   #cordinates
        roi_color=frame[y:y+h,x:x+w]
        
        #creates an image of your face
        id_,conf=recognizer.predict(roi_gray)
        if conf>=50 and conf<=85:
            print(labels[id_])
            font=cv2.FONT_HERSHEY_SIMPLEX
            name=labels[id_]
            color=(0,255,0)
            stroke=2
            cv2.putText(frame,name,(x,y),font,1,color,stroke,cv2.LINE_AA) #font size is 1
       
        
       
        img_item='my-image.png'
        cv2.imwrite(img_item,roi_color) 
        
        #draw a rectangle around your face
        color=(0,0,255)  #BGR
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