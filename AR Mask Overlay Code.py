import cv2

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the mask image
mask = cv2.imread(r"C:\Users\kanan\Desktop\pngegg (1).png", cv2.IMREAD_UNCHANGED)

# Open the webcam (you can change the argument to the camera index if needed)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Overlay the mask on each detected face
    for (x, y, w, h) in faces:
        # Resize the mask to fit the detected face
        resized_mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_AREA)

        # Create a region of interest (ROI) for the face
        roi = frame[y:y + h, x:x + w]

        # Extract the alpha channel from the mask
        mask_alpha = resized_mask[:, :, 3] / 255.0

        # Blend the mask and the face using alpha blending
        for c in range(0, 3):
            roi[:, :, c] = roi[:, :, c] * (1 - mask_alpha) + resized_mask[:, :, c] * mask_alpha

    # Display the frame with face detection and the mask overlay
    cv2.imshow('Face with Mask Overlay', frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
