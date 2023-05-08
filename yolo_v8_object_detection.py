import os
import tkinter as tk
from tkinter import filedialog, simpledialog
from tkinter import ttk
from ultralytics import YOLO
import ultralytics.yolo.data.utils as utils

CONFIDENCE_THRESHOLD = 0.55
IMAGE_FORMATS = utils.IMG_FORMATS
VIDEO_FORMATS = utils.VID_FORMATS

TASK_DICT = {
    "Detection": "",
    "Instance Segmentation": "-seg",
    "Pose/Keypoints": "-pose",
    "Classification": "-cls"
}
SIZE_DICT = {
    "Nano": "n",
    "Small": "s",
    "Medium": "m",
    "Large": "l",
    "Extra Large": "x"
}


class YoloFileProcessor:
    def __init__(self, model_version='yolov8n-seg.pt'):
        self.yolo_model = self._load_model(model_version)

    @staticmethod
    def _load_model(model_version):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        weights_dir = os.path.join(current_dir, 'weights')
        model_path = os.path.join(weights_dir, model_version)
        return YOLO(model_path)

    def process_file(self, file_path):
        return self._predict_with_yolo(file_path)

    def _predict_with_yolo(self, file_path):
        results_gen = self.yolo_model.predict(
            source=file_path,
            conf=CONFIDENCE_THRESHOLD,
            retina_masks=True,
            save=True,
            show=True,
            save_txt=True,
        )
        return results_gen is not None


class CustomUrlDialog(simpledialog.Dialog):
    def body(self, master):
        ttk.Label(master, text="Enter the URL:").grid(row=0)
        self.url_input = ttk.Entry(master, width=50)
        self.url_input.grid(row=0, column=1)
        return self.url_input

    def apply(self):
        self.result = self.url_input.get()


class InputSelectionDialog(tk.Toplevel):
    def __init__(self, parent=None, title=None):
        super().__init__(parent)
        self.title(title)
        self.geometry("300x250")
        self.choice = None

        self.protocol("WM_DELETE_WINDOW", self.quit)
        self.task_var = tk.StringVar(self)
        self.size_var = tk.StringVar(self)
        self.create_input_selection_window()

    def set_choice(self, choice):
        self.choice = choice
        self.destroy()

    def quit(self):
        self.choice = None
        self.destroy()

    def create_input_selection_window(self):
        window = ttk.Frame(self)
        window.pack(padx=20, pady=20)

        file_button = ttk.Button(
            window,
            text="File",
            command=lambda: self.set_choice("file"),
            width=20
        )
        file_button.pack(pady=10)

        url_button = ttk.Button(
            window,
            text="URL",
            command=lambda: self.set_choice("url"),
            width=20
        )
        url_button.pack(pady=10)

        self.task_var.set("Instance Segmentation")
        self.size_var.set("Nano")

        task_menu = ttk.OptionMenu(
            window,
            self.task_var,
            "Instance Segmentation",
            *TASK_DICT.keys())
        task_menu.pack(pady=10)

        size_menu = ttk.OptionMenu(
            window, self.size_var, "Nano", *SIZE_DICT.keys())
        size_menu.pack(pady=10)


def select_input(parent):
    input_dialog = InputSelectionDialog(parent)
    input_dialog.wait_window()

    return input_dialog.choice


def select_file(parent):
    file_path = filedialog.askopenfilename(
        parent=parent,
        filetypes=[
            ('Video files', f"*{';*'.join(VIDEO_FORMATS)}"),
            ('Image files', f"*{';*'.join(IMAGE_FORMATS)}")],
        title='Select a file')
    return file_path


def ask_url(parent):
    url_dialog = CustomUrlDialog(parent=parent, title="Enter URL")
    return url_dialog.result


class YoloApp:
    def __init__(self):
        self.yolo_processor = None

    def process_input(self, parent):
        input_type = select_input(parent)

        if input_type is None:
            print("Exiting...")
            return

        if input_type == "file":
            file_path = select_file(parent)
        else:
            file_path = ask_url(parent)

        if not file_path:
            print("No input provided. Exiting...")
            return

        task_suffix = parent.task_var.get()
        size_suffix = parent.size_var.get()

        model_version = f"yolov8{SIZE_DICT[size_suffix]}{TASK_DICT[task_suffix]}.pt"
        self.yolo_processor = YoloFileProcessor(model_version=model_version)

        print(f"Processing {input_type}: {file_path}")
        is_successful = self.yolo_processor.process_file(file_path)

        if is_successful:
            print(f"Succesfully processed {input_type}")
        else:
            print(f"Failed to process {input_type}")


def main():
    root = tk.Tk()
    root.withdraw()

    app = YoloApp()
    input_choice = app.process_input(root)

    if input_choice is None:
        return
    else:
        root.mainloop()


if __name__ == '__main__':
    main()
