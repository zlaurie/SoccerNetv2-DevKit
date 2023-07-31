import cv2
import numpy as np
import json
from datetime import datetime

VIDEO = "vido"
NUM_METHOD = 4
PRED_FILE = "./Pooling/inference/outputs/Predictions-v2.json"
STR = "Pool : "
font = cv2.FONT_HERSHEY_SIMPLEX

def getTime(time):
    time = datetime.strptime(time, '%M:%S')
    return time.minute * 60 + time.second

pause = False
frame_count = 0
decompte = 0

preds = {}
with open(PRED_FILE) as file:
    preds = json.load(file)["predictions"]

pred = preds[0]
pred_time = getTime(pred["gameTime"][4:])
next_pred = 1

#cap = cv2.VideoCapture(f'C:/Users/Malaurie/Downloads/{VIDEO}.mp4')
cap = cv2.VideoCapture(f'./{VIDEO}.avi')
if (cap.isOpened()== False):
  print("Error opening video stream or file")
fps = cap.get(cv2.CAP_PROP_FPS)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
height = int(NUM_METHOD * frame_height / 6)
out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (frame_width, frame_height))



start_time = datetime.now()
while(cap.isOpened()):
    ret, frame = cap.read()
    frame_count += 1
    if ret == True:
        str = STR
        if pred_time != None and frame_count / fps >= pred_time:
            str += f"{pred['label']} ({int(float(pred['confidence'])*100)}%)"
            decompte += 1
            if decompte > fps*2:
                pred = preds[next_pred] if len(preds) > next_pred else None
                pred_time = getTime(pred["gameTime"][4:]) if len(preds) > next_pred else None
                next_pred += 1
                decompte = 1

        cv2.putText(frame, str, (10, height), font, 1, (255, 255, 255), 2, cv2.LINE_4)
        out.write(frame)

        cv2.imshow('Frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
out.release
cv2.destroyAllWindows()