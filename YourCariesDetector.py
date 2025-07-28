import sys
import pandas as pd
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton,
    QComboBox, QSpinBox, QTextEdit, QMessageBox, QStackedWidget,
    QFileDialog, QTableWidget, QTableWidgetItem, QHBoxLayout, QSizePolicy, QSpacerItem
)
from PySide6.QtCore import QDate, Qt, QTimer
from PySide6.QtGui import QPixmap, QImage
import os
import cv2
from ultralytics import YOLO
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

EXCEL_FILE = "caries_patients.xlsx"
model = YOLO(r"C:\Users\muham\Documents\software_development\best.pt")

def save_patient_data_excel(data):
    new_data = pd.DataFrame([data])
    if os.path.exists(EXCEL_FILE):
        existing_data = pd.read_excel(EXCEL_FILE)
        updated_data = pd.concat([existing_data, new_data], ignore_index=True)
        updated_data.to_excel(EXCEL_FILE, index=False)
    else:
        new_data.to_excel(EXCEL_FILE, index=False)

class HomePage(QWidget):
    def __init__(self, stacked_widget):
        super().__init__()
        self.stacked_widget = stacked_widget
        self.patient_data = {}
        layout = QVBoxLayout()

        title_label = QLabel("🦷 Your Caries Detector")
        title_label.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(title_label)
        
        self.date_edit = QDate.currentDate().toString("yyyy-MM-dd")
        layout.addWidget(QLabel(f"Date: {self.date_edit}"))
        
        self.name_edit = QLineEdit()
        layout.addWidget(QLabel("Patient Name:"))
        layout.addWidget(self.name_edit)
        
        self.gender_combo = QComboBox()
        self.gender_combo.addItems(["Select", "Male", "Female"])
        layout.addWidget(QLabel("Gender:"))
        layout.addWidget(self.gender_combo)
        
        self.age_spin = QSpinBox()
        self.age_spin.setRange(0, 120)
        layout.addWidget(QLabel("Age:"))
        layout.addWidget(self.age_spin)
        
        self.brush_combo = QComboBox()
        self.brush_combo.addItems(["Select", "Once a day", "Twice a day", "Occasionally"])
        layout.addWidget(QLabel("Brushing Habit:"))
        layout.addWidget(self.brush_combo)
        
        self.smoker_combo = QComboBox()
        self.smoker_combo.addItems(["Select", "Smoker", "Non-Smoker"])
        layout.addWidget(QLabel("Smoking Status:"))
        layout.addWidget(self.smoker_combo)
        
        self.dental_combo = QComboBox()
        self.dental_combo.addItems(["Select", "Never", "Less than a year", "More than a year"])
        layout.addWidget(QLabel("Last Dental Appointment:"))
        layout.addWidget(self.dental_combo)
        
        self.notes_edit = QTextEdit()
        layout.addWidget(QLabel("Notes for Dentist:"))
        layout.addWidget(self.notes_edit)
        
        next_button = QPushButton("Next")
        next_button.clicked.connect(self.check_fields)
        layout.addWidget(next_button)
        
        self.setLayout(layout)

    def check_fields(self):
        if (self.name_edit.text() and
            self.gender_combo.currentIndex() != 0 and
            self.brush_combo.currentIndex() != 0 and
            self.smoker_combo.currentIndex() != 0 and
            self.dental_combo.currentIndex() != 0):
            
            self.patient_data = {
                "Date": self.date_edit,
                "Name": self.name_edit.text(),
                "Gender": self.gender_combo.currentText(),
                "Age": self.age_spin.value(),
                "Brushing Habit": self.brush_combo.currentText(),
                "Smoking Status": self.smoker_combo.currentText(),
                "Last Dental Appointment": self.dental_combo.currentText(),
                "Notes": self.notes_edit.toPlainText()
            }
            save_patient_data_excel(self.patient_data)
            QMessageBox.information(self, "Success", "Patient data saved to Excel! Proceeding to Screening.")
            self.stacked_widget.analyzing_page.patient_data = self.patient_data
            self.stacked_widget.setCurrentIndex(1)
        else:
            QMessageBox.warning(self, "Incomplete", "Please fill in all required fields.")

class AnalyzingPage(QWidget):
    def __init__(self, stacked_widget):
        super().__init__()
        self.stacked_widget = stacked_widget
        self.detection_results = []
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.paused = False
        self.mode = None
        self.recording = False
        self.video_writer = None
        self.patient_data = {}

        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignCenter)
        self.setLayout(main_layout)

        title_label = QLabel("🦷 Your Caries Detector - Analyzing")
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; margin-bottom: 20px;")
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)

        self.image_label = QLabel()
        self.image_label.setFixedSize(600, 400)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 2px solid #ccc;")
        main_layout.addWidget(self.image_label, alignment=Qt.AlignCenter)

        main_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)
        main_layout.addLayout(button_layout)

        self.upload_img_btn = QPushButton("📷 Upload Image")
        self.upload_img_btn.clicked.connect(self.upload_image)
        self.upload_img_btn.setStyleSheet(self.button_style())
        button_layout.addWidget(self.upload_img_btn)

        self.remove_img_btn = QPushButton("❌ Remove Image")
        self.remove_img_btn.clicked.connect(self.remove_image)
        self.remove_img_btn.setStyleSheet(self.button_style())
        self.remove_img_btn.setVisible(False)
        button_layout.addWidget(self.remove_img_btn)

        self.upload_video_btn = QPushButton("🎥 Upload Video")
        self.upload_video_btn.clicked.connect(self.upload_video)
        self.upload_video_btn.setStyleSheet(self.button_style())
        button_layout.addWidget(self.upload_video_btn)

        self.open_camera_btn = QPushButton("📸 Open Camera")
        self.open_camera_btn.clicked.connect(self.open_camera)
        self.open_camera_btn.setStyleSheet(self.button_style())
        button_layout.addWidget(self.open_camera_btn)

        self.stop_camera_btn = QPushButton("⏹ Stop Camera")
        self.stop_camera_btn.clicked.connect(self.stop_camera)
        self.stop_camera_btn.setStyleSheet(self.button_style())
        self.stop_camera_btn.setVisible(False)
        button_layout.addWidget(self.stop_camera_btn)

        self.record_btn = QPushButton("⏺ Record")
        self.record_btn.clicked.connect(self.start_recording)
        self.record_btn.setStyleSheet(self.button_style())
        self.record_btn.setVisible(False)
        button_layout.addWidget(self.record_btn)

        self.stop_record_btn = QPushButton("⏹ Stop Recording")
        self.stop_record_btn.clicked.connect(self.stop_recording)
        self.stop_record_btn.setStyleSheet(self.button_style())
        self.stop_record_btn.setVisible(False)
        button_layout.addWidget(self.stop_record_btn)

        self.pause_btn = QPushButton("⏸ Pause")
        self.pause_btn.clicked.connect(self.pause_video)
        self.pause_btn.setStyleSheet(self.button_style())
        self.pause_btn.setVisible(False)
        button_layout.addWidget(self.pause_btn)

        self.continue_btn = QPushButton("▶️ Continue")
        self.continue_btn.clicked.connect(self.continue_video)
        self.continue_btn.setStyleSheet(self.button_style())
        self.continue_btn.setVisible(False)
        button_layout.addWidget(self.continue_btn)

        self.rewind_btn = QPushButton("⏪ Rewind 5s")
        self.rewind_btn.clicked.connect(self.rewind_video)
        self.rewind_btn.setStyleSheet(self.button_style())
        self.rewind_btn.setVisible(False)
        button_layout.addWidget(self.rewind_btn)

        self.stop_video_btn = QPushButton("⏹ Stop Video")
        self.stop_video_btn.clicked.connect(self.stop_video)
        self.stop_video_btn.setStyleSheet(self.button_style())
        self.stop_video_btn.setVisible(False)
        button_layout.addWidget(self.stop_video_btn)

        self.analyze_btn = QPushButton("📊 Analyze")
        self.analyze_btn.clicked.connect(self.go_to_result_page)
        self.analyze_btn.setVisible(False)
        self.analyze_btn.setStyleSheet(self.button_style())
        main_layout.addWidget(self.analyze_btn)

        # Add Back to Home button
        back_btn = QPushButton("⬅️ Back to Home")
        back_btn.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0))
        back_btn.setStyleSheet(self.button_style())
        main_layout.addWidget(back_btn)    

    def button_style(self):
        return """
            QPushButton {
                background-color: #3498db;
                color: white;
                border-radius: 10px;
                padding: 10px 20px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #1c6690;
            }
        """

    def upload_image(self):
        self.clear_video()
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg)")
        if file_path:
            results_list = model(file_path)
            img = results_list[0].plot()
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width, channel = img_rgb.shape
            bytes_per_line = 3 * width
            q_img = QImage(img_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            self.image_label.setPixmap(pixmap)
            self.image_label.setScaledContents(True)
            self.detection_results = [results_list[0]]
            self.mode = "image"
            self.analyze_btn.setVisible(True)
            self.remove_img_btn.setVisible(True)
            self.upload_video_btn.setVisible(False)
            QMessageBox.information(self, "Image Uploaded", "Image uploaded and processed. Click 'Analyze' to continue.")

    def remove_image(self):
        self.image_label.clear()
        self.detection_results = []
        self.mode = None
        self.analyze_btn.setVisible(False)
        self.remove_img_btn.setVisible(False)
        self.upload_video_btn.setVisible(True)

    def upload_video(self):
        self.remove_image()
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Video Files (*.mp4 *.avi *.mov)")
        if file_path:
            self.cap = cv2.VideoCapture(file_path)
            if not self.cap.isOpened():
                QMessageBox.warning(self, "Error", "Failed to open video.")
                return
            self.detection_results = []
            self.mode = "video"
            self.analyze_btn.setVisible(False)
            self.upload_img_btn.setVisible(False)
            self.pause_btn.setVisible(True)
            self.continue_btn.setVisible(True)
            self.rewind_btn.setVisible(True)
            self.stop_video_btn.setVisible(True)
            self.timer.start(30)
            QMessageBox.information(self, "Video Processing", "Video is playing. Analyze button will appear when video is finished.")

    def update_frame(self):
        if self.cap and self.cap.isOpened():
            if not self.paused:
                ret, frame = self.cap.read()
                if ret:
                    results_list = model(frame)
                    self.detection_results.append(results_list[0])
                    img = results_list[0].plot()
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    height, width, channel = img_rgb.shape
                    bytes_per_line = 3 * width
                    q_img = QImage(img_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(q_img)
                    self.image_label.setPixmap(pixmap)
                    self.image_label.setScaledContents(True)
                    if self.recording and self.video_writer:
                        self.video_writer.write(img)
                else:
                    self.cap.release()
                    self.cap = None
                    self.timer.stop()
                    self.analyze_btn.setVisible(True)
                    QMessageBox.information(self, "Video Finished", "Video finished. Click 'Analyze' to view results.")

    def pause_video(self):
        self.paused = True

    def continue_video(self):
        self.paused = False

    def rewind_video(self):
        if self.cap:
            current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            rewind_frames = int(fps * 5)
            target_frame = max(current_frame - rewind_frames, 0)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

    def stop_video(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        self.timer.stop()
        self.image_label.clear()
        self.detection_results = []
        self.mode = None
        self.analyze_btn.setVisible(False)
        for btn in [self.pause_btn, self.continue_btn, self.rewind_btn, self.stop_video_btn]:
            btn.setVisible(False)
        self.upload_img_btn.setVisible(True)
        QMessageBox.information(self, "Video Stopped", "Video playback stopped.")

    def start_recording(self):
        if self.cap and self.patient_data.get("Name"):
            folder_name = f"{self.patient_data['Name']}_result"
            os.makedirs(folder_name, exist_ok=True)
            recording_path = os.path.join(folder_name, "camera_record.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = self.cap.get(cv2.CAP_PROP_FPS) or 20.0
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.video_writer = cv2.VideoWriter(recording_path, fourcc, fps, (width, height))
            self.recording = True
            self.record_btn.setVisible(False)
            self.stop_record_btn.setVisible(True)
            QMessageBox.information(self, "Recording", "Recording started.")

    def stop_recording(self):
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        self.recording = False
        self.record_btn.setVisible(True)
        self.stop_record_btn.setVisible(False)
        QMessageBox.information(self, "Recording Stopped", f"Recording saved to {self.patient_data['Name']}_result/camera_record.mp4")


    def open_camera(self):
        self.remove_image()
        self.clear_video()
        self.cap = cv2.VideoCapture(1)
        if not self.cap.isOpened():
            QMessageBox.warning(self, "Error", "Failed to open camera.")
            return
        self.detection_results = []
        self.mode = "camera"
        self.upload_img_btn.setVisible(False)
        self.upload_video_btn.setVisible(False)
        self.stop_camera_btn.setVisible(True)
        self.record_btn.setVisible(True)
        self.timer.start(30)
        QMessageBox.information(self, "Camera Processing", "Live camera started. Click 'Stop Camera' to finish.")



    def stop_camera(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        self.timer.stop()
        if self.recording and self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            self.recording = False

            # Save the recording into patient's folder
            if self.patient_data.get("Name"):
                folder_name = f"{self.patient_data['Name']}_result"
                os.makedirs(folder_name, exist_ok=True)
                src_file = "camera_record.mp4"
                dst_file = os.path.join(folder_name, "camera_record.mp4")
                if os.path.exists(src_file):
                    os.replace(src_file, dst_file)
                QMessageBox.information(self, "Recording Saved", f"Recording saved as {dst_file}")

        self.analyze_btn.setVisible(True)
        self.stop_camera_btn.setVisible(False)
        self.record_btn.setVisible(False)
        self.stop_record_btn.setVisible(False)
        self.upload_img_btn.setVisible(True)
        self.upload_video_btn.setVisible(True)
        QMessageBox.information(self, "Camera Stopped", "Click 'Analyze' to view results.")

    def clear_video(self):
        if self.cap:
            self.cap.release()
            self.cap = None
            self.timer.stop()
            for btn in [self.pause_btn, self.continue_btn, self.rewind_btn, self.stop_video_btn]:
                btn.setVisible(False)
            self.upload_img_btn.setVisible(True)

    def go_to_result_page(self):
        if self.detection_results:
            result_page = ResultPage(self.stacked_widget, self.detection_results, self.mode, self.patient_data)
            if self.stacked_widget.count() > 2:
                self.stacked_widget.removeWidget(self.stacked_widget.widget(2))
            self.stacked_widget.addWidget(result_page)
            self.stacked_widget.setCurrentWidget(result_page)
        else:
            QMessageBox.warning(self, "No Data", "Please upload, complete video, or stop camera before analyzing.")

class ResultPage(QWidget):
    def __init__(self, stacked_widget, detection_results, mode, patient_data):
        super().__init__()
        self.stacked_widget = stacked_widget

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        self.setLayout(layout)

        title_label = QLabel("📊 Caries Analysis Results")
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; margin-bottom: 20px;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        fig = Figure(figsize=(4, 4))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)

        classes = ["Healthy", "Initial", "Moderate", "Extensive"]
        counts = {cls: 0 for cls in classes}
        for result in detection_results:
            for r in result.boxes.cls.tolist():
                label = model.names[int(r)]
                if label in counts:
                    counts[label] += 1

        total = sum(counts.values()) or 1
        sizes = [counts[cls] for cls in classes]
        labels = [f"{cls} ({(counts[cls]/total)*100:.1f}%)" for cls in classes]

        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
        ax.set_title("Caries Class Distribution")
        layout.addWidget(canvas)

        table = QTableWidget()
        table.setRowCount(len(classes))
        if mode == "image":
            table.setColumnCount(3)
            table.setHorizontalHeaderLabels(["Class", "Quantity", "Percentage (%)"])
            for i, cls in enumerate(classes):
                quantity = counts[cls]
                percentage = (quantity / total) * 100
                table.setItem(i, 0, QTableWidgetItem(cls))
                table.setItem(i, 1, QTableWidgetItem(str(quantity)))
                table.setItem(i, 2, QTableWidgetItem(f"{percentage:.2f}"))
        else:
            table.setColumnCount(2)
            table.setHorizontalHeaderLabels(["Class", "Percentage (%)"])
            for i, cls in enumerate(classes):
                percentage = (counts[cls] / total) * 100
                table.setItem(i, 0, QTableWidgetItem(cls))
                table.setItem(i, 1, QTableWidgetItem(f"{percentage:.2f}"))

        detail_layout = QVBoxLayout()
        for key, value in patient_data.items():
            detail_layout.addWidget(QLabel(f"{key}: {value}"))

        h_layout = QHBoxLayout()
        h_layout.addWidget(table)
        h_layout.addLayout(detail_layout)
        layout.addLayout(h_layout)

        # Save Result Button
        save_btn = QPushButton("💾 Save Analysis Result")
        save_btn.setStyleSheet(self.button_style())
        save_btn.clicked.connect(lambda: self.save_analysis(patient_data["Name"], fig))
        layout.addWidget(save_btn)

        back_btn = QPushButton("⬅️ Back to Analyzing")
        back_btn.setStyleSheet(self.button_style())
        back_btn.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(1))
        layout.addWidget(back_btn)

    def save_analysis(self, patient_name, figure):
        folder_name = f"{patient_name}_result"
        os.makedirs(folder_name, exist_ok=True)
        # Save the entire ResultPage layout
        pixmap = self.grab()
        file_path = os.path.join(folder_name, "analysis_result_fullpage.png")
        pixmap.save(file_path)
        QMessageBox.information(self, "Saved", f"Full analysis page saved as {file_path}")

    def button_style(self):
        return """
            QPushButton {
                background-color: #3498db;
                color: white;
                border-radius: 10px;
                padding: 10px 20px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #1c6690;
            }
        """

class StackedWidget(QStackedWidget):
    def __init__(self):
        super().__init__()
        self.home_page = HomePage(self)
        self.analyzing_page = AnalyzingPage(self)
        self.addWidget(self.home_page)
        self.addWidget(self.analyzing_page)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    stacked_widget = StackedWidget()
    stacked_widget.setFixedSize(1000, 800)
    stacked_widget.show()
    sys.exit(app.exec())
