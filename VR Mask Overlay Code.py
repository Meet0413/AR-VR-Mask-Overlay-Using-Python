import cv2
import tkinter as tk
from tkinter import filedialog

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create a simple Tkinter window for file selection
root = tk.Tk()
root.withdraw()

# Ask the user to select an image file
print("Please select the face image of human:")
file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])

if not file_path:
    print("No file selected. Exiting...")
    exit()

# Print the selected image path
print("Image selected:", file_path)

# Ask the user to select a mask image file
print("Please select the mask image in PNG format:")
mask_path = filedialog.askopenfilename(title="Select Mask Image", filetypes=[("PNG files", "*.png")])

if not mask_path:
    print("No mask selected. Exiting...")
    exit()

# Print the selected mask image path
print("Mask image selected:", mask_path)

# Load the selected image
frame = cv2.imread(file_path)
# Convert the frame to grayscale for face detection
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Detect faces in the frame
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.8, minNeighbors=5)

# Load the mask image with an alpha channel
mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

# Iterate over each detected face
for (x, y, w, h) in faces:
    print("Detected Face Coordinates (x, y, w, h):", x, y, w, h)

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
cv2.imshow('Virtual Faces with Mask Overlay', frame)

# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
