# Sign Language Recognition - Setup Guide

This document provides step-by-step instructions to set up and run the Sign Language Recognition application on a Windows computer.

## System Requirements

- Windows 10 or 11
- Python 3.9+ installed
- Webcam

## Installation Steps

1. **Clone or download the project folder** to your new computer.

2. **Open Command Prompt** (cmd) as administrator:
   - Press `Win + X` and select "Command Prompt (Admin)"
   - Navigate to the project folder: `cd path\to\project\folder`

3. **Create a virtual environment** (recommended):
   ```
   python -m venv venv
   ```

4. **Activate the virtual environment**:
   ```
   venv\Scripts\activate
   ```

5. **Install the required packages**:
   ```
   pip install -r requirements.txt
   ```
   This may take a few minutes as it installs all necessary libraries.

6. **Run the application**:
   ```
   python app.py
   ```

7. **Access the application** by opening a web browser and navigating to:
   ```
   http://127.0.0.1:5001
   ```

## Troubleshooting

If you encounter any issues:

1. **Webcam access problems**:
   - Make sure your webcam is connected and functioning
   - Ensure your browser has permission to access the camera
   - Try closing other applications that might be using the webcam

2. **Package installation errors**:
   - If you see errors about missing DLLs, try installing the Microsoft Visual C++ Redistributable:
     https://aka.ms/vs/17/release/vc_redist.x64.exe
   
   - If mediapipe installation fails, try:
     ```
     pip install mediapipe-silicon
     ```

3. **Performance issues**:
   - If the application is running slowly, try:
     - Ensuring good lighting conditions
     - Using a plain background
     - Closing other resource-intensive applications

## Tips for Best Recognition Results

1. Use good, even lighting (avoid backlighting)
2. Keep your hand about 1.5-2 feet (45-60cm) from the camera
3. Use a plain background when possible
4. Hold signs steady for better recognition
5. Position your hand within the center of the camera frame

## Contact

If you encounter persistent issues, please contact the developer with details of the problem and screenshots if possible. 