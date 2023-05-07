import os
import tkinter as tk
from tkinter import filedialog, simpledialog
from ultralytics import YOLO
import ultralytics.yolo.data.utils as utils

CONFIDENCE_THRESHOLD = 0.55
IMAGE_FORMATS = utils.IMG_FORMATS
VIDEO_FORMATS = utils.VID_FORMATS


class YoloFileProcessor:
    """
    This class is responsible for processing files with YOLO model.
    """

    def __init__(self, model_version='yolov8n-seg.pt'):
        self.yolo_model = self._load_model(model_version)

    @staticmethod
    def _load_model(model_version):
        """
        Load the YOLO model with the given model version.

        :param model_version: str, the version of the YOLO model to load
        :return: YOLO object
        """
        current_dir = os.path.dirname(os.path.realpath(__file__))
        weights_dir = os.path.join(current_dir, 'weights')
        model_path = os.path.join(weights_dir, model_version)
        return YOLO(model_path)

    def process_file(self, file_path):
        """
        Process the given file with the YOLO model.

        :param file_path: str, path of the file to be processed
        :return: bool, True if the file was processed successfully, False otherwise
        """
        return self._predict_with_yolo(file_path)

    def _predict_with_yolo(self, file_path):
        """
        Predict objects in the given file using the YOLO model.

        :param file_path: str, path of the file to predict objects
        :return: bool, True if the prediction was successful, False otherwise
        """
        results_gen = self.yolo_model.predict(
            source=file_path,
            conf=CONFIDENCE_THRESHOLD,
            retina_masks=True,
            save=True,
            show=True,
            save_txt=True,
        )
        return results_gen is not None


class FileSelector:
    """
    This class is responsible for selecting a file or URL using a GUI dialog.
    """

    @staticmethod
    def select_input():
        """
        Show a dialog for the user to select whether to use a file or a URL.

        :return: str, 'file' or 'url' based on the user's choice
        """
        def set_choice(choice):
            user_choice.set(choice)
            root.quit()

        root = tk.Tk()
        root.title("Select Input Type")
        root.geometry("300x100")

        user_choice = tk.StringVar(value="file")

        tk.Radiobutton(
            root,
            text="File",
            variable=user_choice,
            value="file",
            command=lambda: set_choice("file"),
            width=20,
            height=3).pack(
            anchor=tk.W)  # Add width and height
        tk.Radiobutton(
            root,
            text="URL",
            variable=user_choice,
            value="url",
            command=lambda: set_choice("url"),
            width=20,
            height=3).pack(
            anchor=tk.W)

        root.mainloop()
        root.destroy()

        return user_choice.get()

    @staticmethod
    def select_file():
        """
        Open a file selection dialog and return the selected file path.

        :return: str, path of the selected file
        """
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            filetypes=[
                ('Video files', f"*{';*'.join(VIDEO_FORMATS)}"),
                ('Image files', f"*{';*'.join(IMAGE_FORMATS)}")],
            title='Select a file')
        root.destroy()
        return file_path

    @staticmethod
    def ask_url():
        """
        Open a URL input dialog and return the entered URL.

        :return: str, the entered URL
        """
        root = tk.Tk()
        root.withdraw()
        url = simpledialog.askstring("Enter URL", "Enter the URL:")
        root.destroy()
        return url


def process_input():
    """
    Process a file or URL with the YOLO model using a file selector dialog or URL input dialog.
    """
    input_type = FileSelector.select_input()

    if input_type == "file":
        file_path = FileSelector.select_file()
    else:
        file_path = FileSelector.ask_url()

    if not file_path:
        print("No input provided. Exiting...")
        return

    print(f"Processing {input_type}: {file_path}")
    processor = YoloFileProcessor()
    is_successful = processor.process_file(file_path)

    if is_successful:
        print(f"Succesfully processed {input_type}")
    else:
        print(f"Failed to process {input_type}")


def main():
    """
    The main entry point of the script.
    """
    process_input()


if __name__ == '__main__':
    main()
