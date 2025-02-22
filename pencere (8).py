from ultralytics import YOLO
import cv2 as cv

class_name = {
    0:"pencere"
}

model = YOLO("last.pt")

video_path="video.mp4"
cap = cv.VideoCapture(video_path)

screen_width = 1280
screen_height = 720

while cap.isOpened():
    ret , frame = cap.read()
    if not ret : 
        break

    frame =  cv.resize(frame,(screen_width,screen_height))
    results = model (frame)[0]
    for result in results.boxes.data.tolist():
        x1 , y1, x2, y2, score, ID = result
        x1 , y1, x2, y2 = map(int, [x1, y1, x2, y2])

        if score>= 0.5:
            color = (0 , 0 , 255)
            thickness = 2
            cv.rectangle(frame, (x1, y1), (x2,y2), color, thickness)

            if int (ID) in class_name:
                cv.putText(frame, class_name[int(ID)] , (x1, y1+20),cv.FONT_HERSHEY_SIMPLEX, 1,(255,255,0), 2)

    cv.imshow("Video", frame)

    if cv.waitKey(1) & 0xFF== ord('q'):
        break
    





cap.release()
cv.destroyAllWindows()


