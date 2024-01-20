

**Real-Time Face Mask Overlay**

**Overview**

This repository contains a Python script using the OpenCV library to perform real-time face detection from a webcam feed and overlay a mask on the detected faces. The code is designed to demonstrate a simple application of computer vision in augmenting faces with masks, showcasing the capabilities of OpenCV.

**Features**

- Real-time face detection using a pre-trained Haar Cascade classifier.
- Overlaying a mask on detected faces in the webcam feed.
- Adjustable parameters for face detection (scale factor, minimum neighbors).
- Supports any mask with an alpha channel for transparency.

**Prerequisites**

- Python 3.x
- OpenCV (`pip install opencv-python`)

**Usage**



1. Install dependencies:

    
    pip install opencv-python
    

2. Run the script:

    python AR Mask Overlay Code.py
    

3. Quit the application:

    Press 'q' to exit the application.


**Notes**

- Ensure the Haar Cascade XML file for face detection is available in the OpenCV data folder.
- **Adjust the mask image path as needed** (`mask = cv2.imread("path/to/mask.png", cv2.IMREAD_UNCHANGED`).

**Contribution**
Feel free to contribute or provide feedback! Create an issue, open a pull request, or reach out with suggestions.

**License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
