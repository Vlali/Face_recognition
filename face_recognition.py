import cv2
cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    red=cv2.cvtColor(frame,cv2.COLOR_YCR_CB2BGR)
    cv2.imshow('frame',frame)
    cv2.imshow('red',red)
    cv2.imshow('frame3',frame)
    cv2.imshow('frame4',frame)
    cv2.imshow('frame5',frame)
    cv2.imshow('frame6',frame)
    cv2.imshow('frame7',frame)
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()