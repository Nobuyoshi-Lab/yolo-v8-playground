import sys
import importlib
import os
import subprocess
import urllib.request
import csv
from ultralytics.yolo.data.utils import download

REQUIRED_PYTHON_PACKAGES = ['cv2', 'numpy', 'tkinter']
YOLO_VERSIONS_FILE = 'yolo_versions.csv'


def install_package(package):
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", package])
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {package}: {e}")
        sys.exit(1)


def check_python_packages():
    missing_packages = []

    for package in REQUIRED_PYTHON_PACKAGES:
        try:
            importlib.import_module(package)
        except ImportError:
            missing_packages.append(package)

    return missing_packages


def download_yolo_file(file, url):
    try:
        print(f"Downloading {file}...")
        urllib.request.urlretrieve(url, file)
    except Exception as e:
        print(f"Failed to download {file}: {e}")
        sys.exit(1)


def check_yolo_files(version_files):
    missing_files = []

    for file, url in version_files.items():
        if not os.path.exists(file):
            missing_files.append((file, url))

    return missing_files


def read_yolo_versions():
    versions = {}
    with open(YOLO_VERSIONS_FILE, mode='r') as infile:
        reader = csv.reader(infile)
        for row in reader:
            version, file, url = row
            if version not in versions:
                versions[version] = {}
            versions[version][file] = url
    return versions


def download_yolov5_models():
    current_path = os.path.realpath(__file__)
    sys.path.append(os.path.join(current_path, 'utils'))  # add yolov5/ to path

    p5 = list('nsmlx')  # P5 models
    p6 = [f'{x}6' for x in p5]  # P6 models
    cls = [f'{x}-cls' for x in p5]  # classification models
    seg = [f'{x}-seg' for x in p5]  # classification models

    for x in p5 + p6 + cls + seg:
        download.attempt_download(f'weights/yolov5{x}.pt')


def main():
    missing_packages = check_python_packages()

    if missing_packages:
        print(
            f"The following Python packages are missing: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        for package in missing_packages:
            install_package(package)

    yolo_versions = read_yolo_versions()
    selected_version = 'yolov5'  # You can change this to 'yolov5' or 'yolov8'
    version_files = yolo_versions.get(selected_version, {})

    if selected_version == 'yolov5':
        download_yolov5_models()
    else:
        if not version_files:
            print(f"Failed to find files for {selected_version}")
            sys.exit(1)

        missing_files = check_yolo_files(version_files)

        if missing_files:
            print(
                f"The following {selected_version} files are missing: {', '.join(file for file, url in missing_files)}")
            print("Downloading missing files...")
            for file, url in missing_files:
                download_yolo_file(file, url)

    print(f"All prerequisites for {selected_version} are satisfied. You can run the object detection script.")


if __name__ == "__main__":
    main()
