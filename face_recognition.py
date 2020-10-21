import cv2
cap=cv2.VideoCapture(0)



def make_resolution(width,height):
    cap.set(3,width)
    cap.set(4,height)
make_resolution(640,480)  # This is a only resolution,which doesnt crash my program


def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)


while True:
    ret,frame=cap.read()
    green=cv2.cvtColor(frame,cv2.COLOR_YCR_CB2BGR)
    n_frame=rescale_frame(frame,percent=50) #This is your new_frame,which can you rescale or upscale

    cv2.imshow('green',green)
    cv2.imshow('n_frame',n_frame)

    if cv2.waitKey(50) & 0xFF == ord('q'):  # to quit a webcam
        break


cap.release()
cv2.destroyAllWindows()