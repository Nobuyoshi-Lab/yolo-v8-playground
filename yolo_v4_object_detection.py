import cv2
import os
import tkinter as tk
from tkinter import filedialog
from os.path import realpath, dirname
import numpy as np
import glob
import re


# Load YOLOv4 model
def load_yolo_model():
    # Find the latest available YOLO version
    weights_files = glob.glob('yolov*.weights')
    config_files = glob.glob('yolov*.cfg')

    version_numbers = [int(re.search(r'\d+', f).group())
                       for f in weights_files if f.replace('.weights', '.cfg') in config_files]
    if not version_numbers:
        raise ValueError("No valid YOLO models found in the directory.")

    latest_version = max(version_numbers)
    weights_path = f'yolov{latest_version}.weights'
    config_path = f'yolov{latest_version}.cfg'
    classes_path = 'coco.names'

    with open(classes_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    net = cv2.dnn.readNet(weights_path, config_path)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1]
                     for i in net.getUnconnectedOutLayers().flatten()]

    return net, classes, output_layers


# Process the video
def process_video(net, classes, output_layers, video_path):
    cap = cv2.VideoCapture(video_path)
    video_name = os.path.basename(video_path).split('.')[0]
    output_file = os.path.join(
        dirname(
            realpath(__file__)),
        f'{video_name}_output.avi')

    codec = cv2.VideoWriter_fourcc(*'MJPG')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_video = cv2.VideoWriter(output_file, codec, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        blob = cv2.dnn.blobFromImage(
            frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []

        for single_out in outs:
            for detection in single_out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5:
                    center_x, center_y, w, h = map(
                        int, detection[0:4] * np.array([width, height, width, height]))

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out_video.write(frame)

    cap.release()
    out_video.release()
    cv2.destroyAllWindows()


# File dialog for selecting video
def select_video():
    root = tk.Tk()
    root.withdraw()
    video_path = filedialog.askopenfilename(
        filetypes=[
            ('Video files',
             '.mp4;.avi;.mov;.mkv;.flv;.wmv')],
        title='Select a video file')
    return video_path


def main():
    video_path = select_video()

    if not video_path:
        print("No video selected. Exiting...")
        return

    print(f"Processing video: {video_path}")
    net, classes, output_layers = load_yolo_model()
    process_video(net, classes, output_layers, video_path)
    print(f"Processed video saved in the same directory as the script.")


if __name__ == '__main__':
    main()
