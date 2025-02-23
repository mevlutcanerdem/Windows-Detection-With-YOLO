from ultralytics import YOLO
import cv2 as cv

# Define class names for object detection
class_name = {
    0: "window"  # Mapping class ID 0 to "window"
}

# Load the pre-trained YOLO model
model = YOLO("last.pt")

# Define video file path
video_path = "video.mp4"
cap = cv.VideoCapture(video_path)

# Set the desired screen resolution
screen_width = 1280
screen_height = 720

#Get the original video frame rate(default : 30  if unknown)
fps = int(cap.get(cv.CAP_PROP_FPS)) if cap.get(cv.CAP_PROP_FPS) > 0 else 30

#define codec and create a videoWriter object to save output
output_path = "output.mp4"
fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter(output_path,fourcc,fps,(screen_width,screen_height))

# Process the video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Exit loop if there are no more frames

    # Resize the frame to fit the defined screen resolution
    frame = cv.resize(frame, (screen_width, screen_height))
    
    # Perform object detection using the YOLO model
    results = model(frame)[0]
    
    # Iterate through detected objects
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, ID = result  # Extract bounding box and class ID
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])  # Convert coordinates to integers

        # Draw bounding boxes for objects with confidence score >= 0.5
        if score >= 0.5:
            color = (0, 0, 255)  # Red color for bounding box
            thickness = 2
            cv.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # Display class name if it exists in the dictionary
            if int(ID) in class_name:
                cv.putText(frame, class_name[int(ID)], (x1, y1 + 20), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    #Write the frame as output
    out.write(frame)
    # Show the processed video frame
    cv.imshow("Video", frame)

    # Exit loop if 'q' key is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all OpenCV windows
cap.release()
out.release()
cv.destroyAllWindows()
