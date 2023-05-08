# YOLOv8 Playground

This is a simple Python application that uses the YOLOv8 model to perform object detection, instance segmentation, pose estimation, and classification on images and videos. The app uses a graphical user interface (GUI) for input selection and configuration.

## Prerequisites

1. Python 3 installed
2. Install required third-party libraries:

```
pip install -r requirements.txt
```

3. Install the `tkinter` library for the GUI

Linux:

```
sudo apt-get install python3-tk -y
```

For Windows, the `tkinter` library is included in the Python 3 installation.
Use the following command to check if the library is installed:

```
python -m tkinter
```

If the library is not installed, you might need to reinstall Python 3 and select the `tkinter` library during the installation process.

## Setup

1. Clone the repository or download the repository as a ZIP file and extract it.

## How to run

1. Run the `yolo_v8_object_detection.py` file.

```
python yolo_v8_object_detection.py
```

## Usage

1. A window will appear, where you can choose to process a file or provide a URL.
2. Select the task: Detection, Instance Segmentation, Pose/Keypoints, or Classification.
3. Choose the size of the model: Nano, Small, Medium, Large, or Extra Large.
4. Click on "File" or "URL" to select an input source (image or video file, or a URL pointing to an image or video).
5. The app will process the input and display the results. The processed images or videos will be saved in the `runs` folder.

# Credits

This application is based on the [YOLOv8](https://github.com/ultralytics/ultralytics). All credit goes to the original authors. This repository is only a GUI to make it easier to use the model.
