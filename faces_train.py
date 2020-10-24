import cv2
import numpy as np
import os
from PIL import Image
import pickle


BASE_DIR=os.path.dirname(os.path.abspath(__file__))
image_dir=os.path.join(BASE_DIR,'images')


face_cascade=cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_alt2.xml')
recognizer=cv2.face.LBPHFaceRecognizer_create()

current_id=0
label_ids={}
y_labels=[]
x_train=[] # has a pixel values

for root,dirs,files in os.walk(image_dir):
    for file in files:
        if file.endswith('png') or file.endswith('jpg') or file.endswith('jpeg'):
            path=os.path.join(root,file)
            label=os.path.basename(root).replace(' ',"-") #replace its just for you to replace spaces
        #    print(label,path)
            
            if not label in label_ids:  # If a person label is not in the label ids it will set to the dictionary
                label_ids[label]=current_id
                current_id+=1
            id_=label_ids[label]
           # print(label_ids)
         #   y_labels.append(label) #some number
          #  x_train.append(path) #verify this image,turn into a NUMPY array,Gray
            #convert images into numbers
            pil_image=Image.open(path).convert('L') #grayscale     its a python image library
            image_array=np.array(pil_image,'uint8')
          #  print(image_array)
            faces=face_cascade.detectMultiScale(image_array,scaleFactor=1.5,minNeighbors=5)
            
            for(x,y,w,h) in faces:
                roi=image_array[y:y+h,x:x+w]
                x_train.append(roi)
                y_labels.append(id_)

with open('labels.pickle','wb') as f:
    pickle.dump(label_ids,f)

recognizer.train(x_train,np.array(y_labels))
recognizer.save('trainner.yml')

#print(y_labels)
#print(x_train)