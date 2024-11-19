import cv2

from ultralytics import YOLO
from IPython.display import display, Image

def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        print(frame.shape)  
        cv2.imshow("yolov8", frame)

        if(cv2.waitKey(30) == 27):
            break
        

if __name__ == "__main__":
   main()