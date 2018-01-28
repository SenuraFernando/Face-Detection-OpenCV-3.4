
# Import numpy for matrices calculations
import numpy as np

# Import OpenCV2 for image processing
import cv2

# Load prebuilt model for Frontal Face
detector = cv2.CascadeClassifier('HaarCascade/haarcascade_frontalface_default.xml')
# Load prebuilt model for Frontal Eyes
eye_cascade = cv2.CascadeClassifier('HaarCascade/haarcascade_eye.xml')

# Initialize and start the video frame capture
camera = cv2.VideoCapture(0)

# Loop
while (True):

    # Read the video frame
    ret, img = camera.read()

    # Convert the captured frame into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect frames of different sizes, list of faces rectangles
    faces = detector.detectMultiScale(gray, 1.3, 5)

    # For each face in faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 1)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

        # For each eye in faces
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 1)

    # Display the video frame with the bounded rectangle
    cv2.imshow('face', img)

    # If 'q' is pressed, close program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Stop the camera
camera.release()
# Close all windows
cv2.destroyAllWindows()

