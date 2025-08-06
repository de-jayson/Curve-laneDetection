import random

import cv2 as cv
import numpy as np
from ultralytics import YOLO



# opening our dataset file in read mode
my_file = open("C:\\Users\\Ben Alpha\\Python Works\\My Project\\Model\\utils\\coco.txt", "r")

# reading the data/file
data = my_file.read()

# replacing and splitting the text | when newline ("\n") is seen
class_list = data.split("\n")
my_file.close()

# print(class_list)

# Generating random colors for each identified item
detection_colors = []
for i in range(len(class_list)):
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    detection_colors.append((b,g,r))

# Load a pretrained YOLOv8 model
model = YOLO("weights/yolov8n.pt", "v8")

# Values to resize video frames
frame_width = 640
frame_height = 480

# Capturing our loaded images in a video format
# cap = cv.VideoCapture("C:\\Users\\Ben Alpha\\Python Works\\My Project\\inference\\videos\\afriq0.MP4")
cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    #  capture frame by frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True

    if not ret:
        print("Can't receive frame(stream ends?) Exiting .....")
        break

    # Prediction on the image
    
    detect_param = model.predict(source=[frame],conf=0.45, save= False)

    # convert tensor array to numpy
    DP = detect_param[0].numpy()
    print(DP)

    if len(DP) != 0:
        for i in range(len(detect_param[0])):
            print(i)

            boxes = detect_param[0].boxes
            box = boxes[i] #returns one box
            clsID = box.cls.numpy()[0]
            conf = box.conf.numpy()[0]
            bb = box.xyxy.numpy()[0]

            cv.rectangle(
                frame,
                (int(bb[0]),int(bb[1])),
                (int(bb[2]), int(bb[3])),
                detection_colors[int(clsID)],
                3,
            )

            # Display class name and confidence
            font = cv.FONT_HERSHEY_COMPLEX
            cv.putText(
                frame,
                class_list[int(clsID)], 
                # + " " + str(round(conf,3)) + "%",
                (int(bb[0]), int(bb[1]) - 10),
                font,
                1,
                (255,255,255),
                2,
            )
           
    
    # Display the resulting frame
    cv.imshow("ObjectDectector", frame)

    # Terminate run when "Q" is pressed
    if cv.waitKey(1) == ord("q"):
        break

#  When everything done, release capture
cap.release()
cv.destroyAllWindows()