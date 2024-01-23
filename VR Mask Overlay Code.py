
import cv2
import tkinter as tk
from tkinter import filedialog

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create a simple Tkinter window for file selection
root = tk.Tk()
root.withdraw()

# Ask the user to select an image file
file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("All files", "*.jpg;*.jpeg;*.png")])

if not file_path:
    print("No file selected. Exiting...")
    exit()

# Load the mask image
mask = cv2.imread(r"CYour path\pngegg (1).png", cv2.IMREAD_UNCHANGED)

# Open the selected image
frame = cv2.imread(file_path)

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
cv2.imshow('Virtual Face with Mask Overlay', frame)

# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
