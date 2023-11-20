import cv2
import numpy as np
import datetime

# Load YOLO
net = cv2.dnn.readNet("C:\\Users\\Prashik\\Desktop\\Number\\yolov3.weights","C:\\Users\\Prashik\\Desktop\\Number\\yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load COCO names (classes)
with open("C:\\Users\\Prashik\\Desktop\\Number\\coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# List of input video files
video_files = ["C:\\Users\\Prashik\\Desktop\\Number\\Camera Parkng.mp4","C:\\Users\\Prashik\\Desktop\\Number\\parking1.mp4"]
# Initial video index
current_video_index = 0

# Get total number of parking lots
total_lots = 20
# Initialize taken lots counter
taken_lots = 0

# Open a window titled "Parking lot"
cv2.namedWindow("Parking lot", cv2.WINDOW_NORMAL)

# Initialize VideoCapture object outside the loop
cap = cv2.VideoCapture(video_files[current_video_index])

# Create font object for LEMON MILK BOLD font
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1.5
font_thickness = 2

while True:
    # Read frame from the current video file
    ret, frame = cap.read()

    # Stop the loop if the video is finished
    if not ret:
        break

    # Get frame dimensions
    height, width, channels = frame.shape

    # Detecting objects (cars and trucks) on the frame
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Information to draw bounding boxes
    class_ids = []
    confidences = []
    boxes = []

    # Process each output layer
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5 and (class_id == 2 or class_id == 7):  # Class ID for cars is 2, for trucks is 7
                # Object detected: Get coordinates of the bounding box
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Add information to lists for drawing bounding boxes
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-maximum suppression to remove duplicate and low-confidence bounding boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Count taken lots based on detected cars and trucks
    taken_lots = len(indexes)

    # Draw bounding boxes and labels on the frame
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = classes[class_ids[i]]
            confidence = confidences[i]
            color = (0, 255, 0)  # BGR color for car bounding box (green in this case)
            if label == "truck":
                color = (0, 165, 255)  # BGR color for truck bounding box (orange in this case)
                label = "Truck"
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, "{} {:.2f}".format(label, confidence), (x, y - 10), font, font_scale, color, font_thickness, lineType=cv2.LINE_AA)

    # Overlay text showing parking lot information
    text_overlay = "Total Lots = {}\nTaken Lots = {}\nEmpty Lots = {}".format(total_lots, taken_lots, total_lots - taken_lots)
    cv2.putText(frame, text_overlay, (20, 40), font, font_scale, (255, 255, 255), font_thickness, lineType=cv2.LINE_AA)

    # Get size of the extra text
    extra_text = "Extra Text Example"
    extra_text_size = cv2.getTextSize(extra_text, font, font_scale, font_thickness)[0]

    # Get current date and time and draw it just below the extra text in black color and LEMON MILK BOLD font
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    date_time_text = "Date & Time: " + current_datetime
    date_time_text_size = cv2.getTextSize(date_time_text, font, font_scale, font_thickness)[0]
    date_time_x = (width - date_time_text_size[0]) // 2  # Centered horizontally
    date_time_y = extra_text_size[1] + 60  # Positioned just below the extra text
    cv2.putText(frame, date_time_text, (date_time_x, date_time_y), font, font_scale, (0, 0, 0), font_thickness, lineType=cv2.LINE_AA)

    # Display the resulting frame with bounding boxes, overlay text, extra text, and current date and time in the "Parking lot" window
    cv2.imshow("Parking lot", frame)

    # Wait for the 'n' key and transition to the next video
    key = cv2.waitKey(1) & 0xFF
    if key == ord("n"):
        current_video_index = (current_video_index + 1) % len(video_files)
        # Release the previous VideoCapture object and open the new video file
        cap.release()
        cap = cv2.VideoCapture(video_files[current_video_index])

    # Break the loop if 'q' key is pressed
    if key == ord("q"):
        break

# Release the VideoCapture object and close windows
cap.release()
cv2.destroyAllWindows()
