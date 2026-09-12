from pathlib import Path
import sys

import cv2
import numpy as np
from PyQt5 import QtCore as qtc
from PyQt5 import QtGui as qtg
from PyQt5 import QtWidgets as qtw

from deeplearning import face_mask_prediction


BASE_DIR = Path(__file__).resolve().parent


class VideoCapture(qtc.QThread):
    change_pixmap_signal = qtc.pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.run_flag = True

    def run(self):
        # Use the default local camera.
        cap = cv2.VideoCapture(0)

        while self.run_flag:
            ret, frame = cap.read()

            if not ret:
                self.msleep(30)
                continue

            prediction_img = face_mask_prediction(frame)
            self.change_pixmap_signal.emit(prediction_img)
            self.msleep(10)

        cap.release()

    def stop(self):
        self.run_flag = False
        self.wait()


class MainWindow(qtw.QWidget):
    def __init__(self):
        super().__init__()

        icon_path = BASE_DIR / "images" / "icon.png"
        if icon_path.exists():
            self.setWindowIcon(qtg.QIcon(str(icon_path)))

        self.setWindowTitle("Face Mask Detection")
        self.setFixedSize(600, 600)

        title = qtw.QLabel("<h2>Face Mask Detection</h2>")

        self.camera_button = qtw.QPushButton(
            "Open Camera",
            clicked=self.camera_button_click,
            checkable=True,
        )

        self.screen = qtw.QLabel()
        placeholder = qtg.QPixmap(600, 480)
        placeholder.fill(qtg.QColor("darkGray"))
        self.screen.setPixmap(placeholder)

        layout = qtw.QVBoxLayout()
        layout.addWidget(title)
        layout.addWidget(self.camera_button)
        layout.addWidget(self.screen)

        self.setLayout(layout)

    def camera_button_click(self):
        if self.camera_button.isChecked():
            self.camera_button.setText("Close Camera")

            self.capture = VideoCapture()
            self.capture.change_pixmap_signal.connect(self.update_image)
            self.capture.start()
        else:
            self.camera_button.setText("Open Camera")

            if hasattr(self, "capture"):
                self.capture.stop()

    @qtc.pyqtSlot(np.ndarray)
    def update_image(self, image_array):
        rgb_img = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

        height, width, channels = rgb_img.shape
        bytes_per_line = channels * width

        converted_image = qtg.QImage(
            rgb_img.data,
            width,
            height,
            bytes_per_line,
            qtg.QImage.Format_RGB888,
        )

        scaled_image = converted_image.scaled(
            600,
            480,
            qtc.Qt.KeepAspectRatio,
        )

        self.screen.setPixmap(qtg.QPixmap.fromImage(scaled_image))

    def closeEvent(self, event):
        if hasattr(self, "capture"):
            self.capture.stop()

        event.accept()


if __name__ == "__main__":
    app = qtw.QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())
