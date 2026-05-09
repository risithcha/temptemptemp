import sys
import requests
import cv2
import numpy as np
import base64
import threading
import winsound
import socketio as socketio_client
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, 
    QHBoxLayout, QPushButton, QLabel, QStackedWidget,
    QTextEdit, QGroupBox, QSplitter, QFrame
)
from PyQt6.QtCore import Qt, QTimer, QSize, pyqtSignal, QObject
from PyQt6.QtGui import QPalette, QColor, QImage, QPixmap, QFont
from data_overlay import OverlayLabel
from audio_engine import AudioRecorder


class StatusIndicator(QWidget):
    """A small circular status indicator widget."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.status = "disconnected"
        self.setFixedSize(20, 20)
        self.setToolTip("Checking backend connection...")
        
    def set_connected(self):
        """Set the indicator to connected state (green)."""
        self.status = "connected"
        self.setToolTip("Connected")
        self.update()
        
    def set_disconnected(self, error_message="Backend not running"):
        """Set the indicator to disconnected state (red)."""
        self.status = "disconnected"
        self.setToolTip(f"Error: {error_message}")
        self.update()
        
    def paintEvent(self, event):
        """Paint the status indicator circle."""
        from PyQt6.QtGui import QPainter
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        if self.status == "connected":
            painter.setBrush(QColor(76, 175, 80))  # Green
        else:
            painter.setBrush(QColor(244, 67, 54))  # Red
            
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(2, 2, 16, 16)


class ModeSelectionView(QWidget):
    """The initial view with mode selection buttons."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(30)
        
        # Title
        title = QLabel("Welcome to Atlas")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("""
            font-size: 32px;
            font-weight: bold;
            color: #333;
            margin-bottom: 20px;
        """)
        layout.addWidget(title)
        
        # Subtitle
        subtitle = QLabel("Select an Assist Mode")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("""
            font-size: 18px;
            color: #666;
            margin-bottom: 30px;
        """)
        layout.addWidget(subtitle)
        
        # Vision Assist Mode Button
        self.vision_button = QPushButton("Vision Assist Mode")
        self.vision_button.setMinimumSize(QSize(300, 80))
        self.vision_button.setStyleSheet("""
            QPushButton {
                font-size: 20px;
                font-weight: bold;
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 10px;
                padding: 20px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
        """)
        layout.addWidget(self.vision_button)
        
        # Hearing Assist Mode Button
        self.hearing_button = QPushButton("Hearing Assist Mode")
        self.hearing_button.setMinimumSize(QSize(300, 80))
        self.hearing_button.setStyleSheet("""
            QPushButton {
                font-size: 20px;
                font-weight: bold;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 10px;
                padding: 20px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #2E7D32;
            }
        """)
        layout.addWidget(self.hearing_button)
        
        layout.addStretch()
        self.setLayout(layout)


class VisionModeWidget(QWidget):
    """Widget for Vision Assist Mode with live webcam feed."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.camera = None
        self.timer = None
        self.analysis_timer = None
        self.current_frame = None
        self.backend_url = "http://127.0.0.1:5000"
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface."""
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Set background color
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(0, 0, 0))
        self.setPalette(palette)
        
        # Create a splitter for resizable layout between video and sidebar
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setStyleSheet("QSplitter::handle { background-color: #333; }")
        
        # Video container widget
        video_container = QWidget()
        video_container.setStyleSheet("background-color: black;")
        video_layout = QVBoxLayout(video_container)
        video_layout.setContentsMargins(0, 0, 0, 0)
        video_layout.setSpacing(0)
        
        # Video display label with overlay capability
        self.video_label = OverlayLabel(self)
        # Center both horizontally and vertically
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setScaledContents(False)  # Disable to prevent distortion for object detection
        self.video_label.setMinimumSize(480, 360)  # Ensure minimum visibility
        video_layout.addWidget(self.video_label, 1)  # Stretch factor 1 for maximum space
        
        # Add video container to splitter
        self.splitter.addWidget(video_container)
        
        # Create sidebar for text information
        sidebar = QWidget()
        sidebar.setMinimumWidth(250)
        sidebar.setMaximumWidth(400)
        sidebar.setStyleSheet("background-color: #1a1a1a;")
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(10, 10, 10, 10)
        sidebar_layout.setSpacing(10)
        
        # OCR Text Group
        ocr_group = QGroupBox("Detected Text (OCR)")
        ocr_group.setStyleSheet("""
            QGroupBox {
                font-size: 14px;
                font-weight: bold;
                color: #4CAF50;
                border: 1px solid #4CAF50;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        ocr_layout = QVBoxLayout(ocr_group)
        
        self.ocr_text_display = QTextEdit()
        self.ocr_text_display.setReadOnly(True)
        self.ocr_text_display.setPlaceholderText("No text detected yet...")
        self.ocr_text_display.setStyleSheet("""
            QTextEdit {
                background-color: #2d2d2d;
                color: #e0e0e0;
                border: none;
                border-radius: 5px;
                padding: 8px;
                font-size: 13px;
            }
        """)
        self.ocr_text_display.setMinimumHeight(120)
        ocr_layout.addWidget(self.ocr_text_display)
        
        sidebar_layout.addWidget(ocr_group)
        
        # Scene Description Group
        scene_group = QGroupBox("Scene Description")
        scene_group.setStyleSheet("""
            QGroupBox {
                font-size: 14px;
                font-weight: bold;
                color: #2196F3;
                border: 1px solid #2196F3;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        scene_layout = QVBoxLayout(scene_group)
        
        self.scene_description_display = QTextEdit()
        self.scene_description_display.setReadOnly(True)
        self.scene_description_display.setPlaceholderText("Analyzing scene...")
        self.scene_description_display.setStyleSheet("""
            QTextEdit {
                background-color: #2d2d2d;
                color: #e0e0e0;
                border: none;
                border-radius: 5px;
                padding: 8px;
                font-size: 13px;
            }
        """)
        self.scene_description_display.setMinimumHeight(150)
        scene_layout.addWidget(self.scene_description_display)
        
        sidebar_layout.addWidget(scene_group)
        sidebar_layout.addStretch()  # Push content to top
        
        # Add sidebar to splitter
        self.splitter.addWidget(sidebar)
        
        # Set initial splitter sizes (70% video, 30% sidebar)
        self.splitter.setSizes([700, 300])
        self.splitter.setStretchFactor(0, 7)
        self.splitter.setStretchFactor(1, 3)
        
        main_layout.addWidget(self.splitter, 1)
        
        # Back button container
        button_container = QWidget()
        button_container.setStyleSheet("background-color: rgba(0, 0, 0, 180);")
        button_layout = QHBoxLayout(button_container)
        button_layout.setContentsMargins(20, 10, 20, 10)
        
        # Back button
        self.back_button = QPushButton("Back to Mode Selection")
        self.back_button.setMinimumSize(QSize(250, 50))
        self.back_button.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        button_layout.addStretch()
        button_layout.addWidget(self.back_button)
        button_layout.addStretch()
        
        main_layout.addWidget(button_container)
        self.setLayout(main_layout)
        
    def start_camera(self):
        """Start the webcam feed."""
        if self.camera is None:
            # Use DirectShow backend instead of MSMF to avoid frame grab errors
            self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            
            if not self.camera.isOpened():
                self.video_label.setText("Error: Could not access webcam")
                self.video_label.setStyleSheet("""
                    color: white;
                    font-size: 24px;
                    background-color: black;
                """)
                return
            
            # Set camera properties for better performance
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            # Set buffer size to 1 to minimize lag
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
        if self.timer is None:
            self.timer = QTimer()
            self.timer.timeout.connect(self.update_frame)
            self.timer.start(33)
        
        # Start analysis timer for backend communication
        if self.analysis_timer is None:
            self.analysis_timer = QTimer()
            self.analysis_timer.timeout.connect(self.send_frame_for_analysis)
            self.analysis_timer.start(1500)  # Send frame every 1.5 seconds 
            
    def stop_camera(self):
        """Stop the webcam feed."""
        if self.timer is not None:
            self.timer.stop()
            self.timer = None
        
        if self.analysis_timer is not None:
            self.analysis_timer.stop()
            self.analysis_timer = None
            
        if self.camera is not None:
            self.camera.release()
            self.camera = None
            
        self.video_label.clear()
        
    def update_frame(self):
        """Capture and display a frame from the webcam."""
        if self.camera is None or not self.camera.isOpened():
            return
            
        ret, frame = self.camera.read()
        if ret:
            # Store current frame for backend analysis
            self.current_frame = frame
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Get frame dimensions
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            
            # Convert to QImage
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            
            # Scale while maintaining aspect ratio
            scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                self.video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            
            self.video_label.setPixmap(scaled_pixmap)
    
    def send_frame_for_analysis(self):
        """Send the current frame to backend for object detection analysis."""
        if self.current_frame is None:
            return
        
        try:
            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', self.current_frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Send POST request to backend
            response = requests.post(
                f"{self.backend_url}/vision",
                json={"image": frame_base64},
                timeout=2
            )
            
            if response.status_code == 200:
                response_data = response.json()
                
                # Update overlay with detection data and frame size for coordinate mapping
                frame_height, frame_width = self.current_frame.shape[:2]
                self.video_label.update_data(response_data, frame_size=(frame_width, frame_height))
                
                # Update OCR text display
                ocr_text = response_data.get('ocr_text', '')
                if ocr_text and ocr_text.strip():
                    self.ocr_text_display.setPlainText(ocr_text)
                else:
                    self.ocr_text_display.setPlainText("No text detected in current view.")
                
                # Update scene description display
                scene_description = response_data.get('scene_description', '')
                if scene_description and scene_description.strip():
                    self.scene_description_display.setPlainText(scene_description)
                else:
                    # If no scene description, create a basic one from detected objects
                    detections = response_data.get('detections', [])
                    if detections:
                        object_names = [d.get('label', 'object') for d in detections]
                        unique_objects = list(set(object_names))
                        basic_description = f"Detected objects: {', '.join(unique_objects)}"
                        self.scene_description_display.setPlainText(basic_description)
                    else:
                        self.scene_description_display.setPlainText("Analyzing scene...")
        
        except requests.exceptions.RequestException as e:
            # Silently handle backend communication errors
            pass
    
    def showEvent(self, event):
        """Called when widget is shown."""
        super().showEvent(event)
        self.start_camera()
        
    def hideEvent(self, event):
        """Called when widget is hidden."""
        super().hideEvent(event)
        self.stop_camera()


class HearingModeWidget(QWidget):
    """Hearing Assist widget with live captions and threaded audio capture."""
    
    # Signals for thread-safe UI updates
    transcription_received = pyqtSignal(str, bool)  # (text, is_final)
    error_received = pyqtSignal(str)          # Emitted when an error occurs
    status_update = pyqtSignal()              # Emitted to update status to listening
    alert_received = pyqtSignal(str)          # Emitted when a safety alert is detected
    alert_cleared = pyqtSignal()              # Emitted when alert condition clears
    
    # Safe flash interval (500ms = 1Hz, well below 3Hz epilepsy threshold)
    ALERT_FLASH_INTERVAL_MS = 500
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.backend_url = "http://127.0.0.1:5000"
        self.recorder = None
        self.is_listening = False
        self.alert_flash_timer = None
        self.alert_flash_state = False
        self.current_alert = None
        self._live_text = ""       # Current interim preview text
        self._history_text = ""    # Finalized/committed transcription text
        
        # SocketIO client for real-time streaming
        self.sio = socketio_client.Client(reconnection=True, reconnection_delay=1)
        self._setup_socketio()
        
        # Connect signals to slots (thread-safe)
        self.transcription_received.connect(self._update_live_transcription)
        self.error_received.connect(self._show_error)
        self.status_update.connect(self._update_status_listening)
        self.alert_received.connect(self._show_alert_overlay)
        self.alert_cleared.connect(self._hide_alert_overlay)
        
        self.init_ui()
        self.init_audio_recorder()
        
    def init_ui(self):
        """Initialize the user interface."""
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)
        
        # Set dark background for high contrast
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(26, 26, 26))
        self.setPalette(palette)
        
        # Header with title and status
        header_layout = QHBoxLayout()
        
        # Title
        title = QLabel("Hearing Assist Mode")
        title.setAlignment(Qt.AlignmentFlag.AlignLeft)
        title.setStyleSheet("""
            font-size: 28px;
            font-weight: bold;
            color: #4CAF50;
            padding: 10px;
        """)
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        # Listening status indicator
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.status_label.setStyleSheet("""
            font-size: 16px;
            font-weight: bold;
            color: #888;
            padding: 10px;
        """)
        header_layout.addWidget(self.status_label)
        
        main_layout.addLayout(header_layout)
        
        # Instructions label
        self.instructions_label = QLabel("Click 'Start Listening' and speak clearly. Your speech will appear below.")
        self.instructions_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.instructions_label.setWordWrap(True)
        self.instructions_label.setStyleSheet("""
            font-size: 14px;
            color: #aaa;
            padding: 5px;
            margin-bottom: 10px;
        """)
        main_layout.addWidget(self.instructions_label)
        
        # Main caption display area - large, high contrast, scrollable
        caption_group = QGroupBox("Live Captions")
        caption_group.setStyleSheet("""
            QGroupBox {
                font-size: 16px;
                font-weight: bold;
                color: #4CAF50;
                border: 2px solid #4CAF50;
                border-radius: 10px;
                margin-top: 15px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 10px;
            }
        """)
        caption_layout = QVBoxLayout(caption_group)
        
        self.caption_display = QTextEdit()
        self.caption_display.setReadOnly(True)
        self.caption_display.setPlaceholderText(
            "Your transcribed speech will appear here...\n\n"
            "Tips:\n"
            "- Speak clearly and at a normal pace\n"
            "- Reduce background noise for best results\n"
            "- Each chunk of speech will be transcribed after a few seconds"
        )
        self.caption_display.setStyleSheet("""
            QTextEdit {
                background-color: #1a1a1a;
                color: #ffffff;
                border: none;
                border-radius: 8px;
                padding: 15px;
                font-size: 22px;
                font-family: 'Segoe UI', Arial, sans-serif;
                line-height: 1.5;
            }
            QTextEdit:focus {
                border: 1px solid #4CAF50;
            }
        """)
        self.caption_display.setMinimumHeight(300)
        caption_layout.addWidget(self.caption_display)
        
        main_layout.addWidget(caption_group, 1)  # Stretch factor 1
        
        # ===== CRISIS MODE ALERT OVERLAY =====
        # Hidden overlay that appears on top of everything when an alert is detected
        self.alert_overlay = QFrame(self)
        self.alert_overlay.setObjectName("alertOverlay")
        self.alert_overlay.setStyleSheet("""
            QFrame#alertOverlay {
                background-color: #D32F2F;
                border: 4px solid #FFEB3B;
                border-radius: 15px;
            }
        """)
        self.alert_overlay.setVisible(False)
        
        # Alert overlay layout
        alert_layout = QVBoxLayout(self.alert_overlay)
        alert_layout.setContentsMargins(30, 20, 30, 20)
        alert_layout.setSpacing(10)
        
        # Warning icon and text
        self.alert_icon_label = QLabel("🚨")
        self.alert_icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.alert_icon_label.setStyleSheet("""
            font-size: 64px;
            background: transparent;
        """)
        alert_layout.addWidget(self.alert_icon_label)
        
        self.alert_text_label = QLabel("⚠️ ALARM DETECTED ⚠️")
        self.alert_text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.alert_text_label.setWordWrap(True)
        alert_font = QFont("Segoe UI", 28, QFont.Weight.Bold)
        self.alert_text_label.setFont(alert_font)
        self.alert_text_label.setStyleSheet("""
            color: #FFFFFF;
            background: transparent;
            padding: 10px;
        """)
        alert_layout.addWidget(self.alert_text_label)
        
        # Detailed alert message
        self.alert_detail_label = QLabel("")
        self.alert_detail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.alert_detail_label.setWordWrap(True)
        self.alert_detail_label.setStyleSheet("""
            font-size: 18px;
            color: #FFEB3B;
            background: transparent;
            padding: 5px;
        """)
        alert_layout.addWidget(self.alert_detail_label)
        
        # Dismiss button
        self.alert_dismiss_btn = QPushButton("DISMISS ALERT")
        self.alert_dismiss_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.alert_dismiss_btn.setMinimumSize(200, 50)
        self.alert_dismiss_btn.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                font-weight: bold;
                background-color: #FFEB3B;
                color: #000000;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
            }
            QPushButton:hover {
                background-color: #FFF176;
            }
            QPushButton:pressed {
                background-color: #FFD54F;
            }
        """)
        self.alert_dismiss_btn.clicked.connect(self._dismiss_alert)
        alert_layout.addWidget(self.alert_dismiss_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        
        # Initialize flash timer for alert
        self.alert_flash_timer = QTimer(self)
        self.alert_flash_timer.timeout.connect(self._toggle_alert_flash)
        # ===== END CRISIS MODE OVERLAY =====
        
        # Control buttons container
        button_container = QWidget()
        button_container.setStyleSheet("background-color: transparent;")
        button_layout = QHBoxLayout(button_container)
        button_layout.setContentsMargins(0, 10, 0, 10)
        button_layout.setSpacing(20)
        
        # Start/Stop Listening button
        self.listen_button = QPushButton("Start Listening")
        self.listen_button.setMinimumSize(QSize(250, 60))
        self.listen_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.listen_button.setStyleSheet("""
            QPushButton {
                font-size: 18px;
                font-weight: bold;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 10px;
                padding: 15px 30px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #2E7D32;
            }
            QPushButton:disabled {
                background-color: #666;
                color: #aaa;
            }
        """)
        self.listen_button.clicked.connect(self.toggle_listening)
        button_layout.addStretch()
        button_layout.addWidget(self.listen_button)
        
        # Clear button
        self.clear_button = QPushButton("Clear")
        self.clear_button.setMinimumSize(QSize(120, 60))
        self.clear_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.clear_button.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                font-weight: bold;
                background-color: #555;
                color: white;
                border: none;
                border-radius: 10px;
                padding: 15px 20px;
            }
            QPushButton:hover {
                background-color: #666;
            }
            QPushButton:pressed {
                background-color: #444;
            }
        """)
        self.clear_button.clicked.connect(self.clear_captions)
        button_layout.addWidget(self.clear_button)
        button_layout.addStretch()
        
        main_layout.addWidget(button_container)
        
        # Back button
        self.back_button = QPushButton("Back to Mode Selection")
        self.back_button.setMinimumSize(QSize(250, 50))
        self.back_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.back_button.setStyleSheet("""
            QPushButton {
                font-size: 14px;
                background-color: #333;
                color: #ccc;
                border: 1px solid #555;
                border-radius: 8px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #444;
                color: white;
            }
        """)
        
        back_layout = QHBoxLayout()
        back_layout.addWidget(self.back_button)
        back_layout.addStretch()
        main_layout.addLayout(back_layout)
        
        self.setLayout(main_layout)
        
    
    def init_audio_recorder(self):
        """Initialize the audio recorder and connect signals."""
        self.recorder = AudioRecorder(self)
        # Connect signals
        self.recorder.audio_ready.connect(self.on_audio_ready)
        self.recorder.error_occurred.connect(self.on_audio_error)
        self.recorder.recording_started.connect(self.on_recording_started)
        self.recorder.recording_stopped.connect(self.on_recording_stopped)
    
    def _setup_socketio(self):
        """Wire up SocketIO event handlers for live streaming."""
        @self.sio.on('transcription')
        def _on_transcription(data):
            text = data.get('text', '').strip()
            is_final = data.get('final', False)
            if text:
                self.transcription_received.emit(text, is_final)

        @self.sio.on('classification')
        def _on_classification(data):
            alert = data.get('alert')
            if alert is not None:
                self.alert_received.emit(str(alert))
            elif self.current_alert is not None:
                self.alert_cleared.emit()

        @self.sio.on('connect')
        def _on_connect():
            print("[WS] Connected to backend")

        @self.sio.on('disconnect')
        def _on_disconnect():
            print("[WS] Disconnected from backend")

    def toggle_listening(self):
        """Toggle between listening and stopped states."""
        if self.is_listening:
            self.stop_listening()
        else:
            self.start_listening()
    
    def start_listening(self):
        """Start capturing audio from the microphone."""
        if self.recorder is None:
            self.on_audio_error("Audio recorder not initialized")
            return

        # Connect WebSocket to backend
        if not self.sio.connected:
            try:
                self.sio.connect(self.backend_url, wait_timeout=5)
            except Exception as e:
                print(f"[WS] Connection failed, will use HTTP fallback: {e}")

        # Start recording with 0.5-second chunks for live streaming
        success = self.recorder.start_listening(chunk_duration=0.5)
        
        if success:
            self.is_listening = True
            self.listen_button.setText("Stop Listening")
            self.listen_button.setStyleSheet("""
                QPushButton {
                    font-size: 18px;
                    font-weight: bold;
                    background-color: #f44336;
                    color: white;
                    border: none;
                    border-radius: 10px;
                    padding: 15px 30px;
                }
                QPushButton:hover {
                    background-color: #d32f2f;
                }
                QPushButton:pressed {
                    background-color: #b71c1c;
                }
            """)
    
    def stop_listening(self):
        """Stop capturing audio."""
        if self.recorder:
            self.recorder.stop_listening()
        
        # Disconnect WebSocket
        if self.sio.connected:
            try:
                self.sio.disconnect()
            except Exception:
                pass
        
        # Finalize any remaining live text into history
        if self._live_text.strip():
            if self._history_text:
                self._history_text += " " + self._live_text.strip()
            else:
                self._history_text = self._live_text.strip()
            self._live_text = ""
            self.caption_display.setPlainText(self._history_text)
        
        self.is_listening = False
        self.listen_button.setText("Start Listening")
        self.listen_button.setStyleSheet("""
            QPushButton {
                font-size: 18px;
                font-weight: bold;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 10px;
                padding: 15px 30px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #2E7D32;
            }
        """)
    
    def on_audio_ready(self, audio_data: bytes):
        """
        Handle audio data when a chunk is ready.
        Sends via WebSocket for live streaming, falls back to HTTP.
        """
        if self.sio.connected:
            # Stream via WebSocket (non-blocking, handled by socketio client thread)
            try:
                audio_b64 = base64.b64encode(audio_data).decode('utf-8')
                self.sio.emit('audio_chunk', {'audio': audio_b64})
                return
            except Exception as e:
                print(f"[WS] emit failed, falling back to HTTP: {e}")

        # HTTP fallback
        self.status_label.setText("Processing...")
        self.status_label.setStyleSheet("""
            font-size: 16px;
            font-weight: bold;
            color: #FFC107;
            padding: 10px;
        """)
        thread = threading.Thread(
            target=self._send_audio_to_backend,
            args=(audio_data,),
            daemon=True
        )
        thread.start()
    
    def _send_audio_to_backend(self, audio_data: bytes):
        """Send audio data to the backend for transcription (runs in thread)."""
        try:
            # Encode audio as base64
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            
            print(f"[DEBUG] Sending {len(audio_data)} bytes to backend...")
            
            # Send POST request to backend
            response = requests.post(
                f"{self.backend_url}/hearing",
                json={"audio": audio_base64},
                timeout=30  # Whisper can take time for longer audio
            )
            
            print(f"[DEBUG] Response status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                transcription = data.get('transcription', '')
                
                # ===== CRISIS MODE: Check for safety alerts =====
                alert = data.get('alert')
                if alert is not None:
                    # Safety alert detected! Trigger crisis mode
                    print(f"[CRISIS MODE] Alert received from backend: {alert}")
                    self.alert_received.emit(str(alert))
                else:
                    # No alert, clear any existing alert if present
                    if self.current_alert is not None:
                        self.alert_cleared.emit()
                # ===== END CRISIS MODE CHECK =====
                
                print(f"[DEBUG] Transcription received: '{transcription}'")
                
                if transcription.strip():
                    # Emit signal to safely update UI from background thread
                    print(f"[DEBUG] Emitting transcription signal...")
                    self.transcription_received.emit(transcription.strip(), True)
                else:
                    print("[DEBUG] Empty transcription, updating status...")
                    self.status_update.emit()
            else:
                error_msg = response.json().get('error', 'Unknown error')
                print(f"[DEBUG] Backend error: {error_msg}")
                self.error_received.emit(f"Backend error: {error_msg}")
                
        except requests.exceptions.ConnectionError:
            print("[DEBUG] Connection error to backend")
            self.error_received.emit("Cannot connect to backend")
        except requests.exceptions.Timeout:
            print("[DEBUG] Request timeout")
            self.error_received.emit("Request timed out")
        except Exception as e:
            print(f"[DEBUG] Exception: {e}")
            self.error_received.emit(str(e))
    
    def _update_live_transcription(self, text: str, is_final: bool):
        """Update the caption display with interim or finalized text.
        
        interim (final=False): updates the 'live preview' line at the bottom
        final  (final=True):   commits the text permanently to history
        """
        if is_final:
            # Commit text to permanent history
            if self._history_text:
                self._history_text += " " + text
            else:
                self._history_text = text
            self._live_text = ""
            display = self._history_text
        else:
            # Update interim live preview
            self._live_text = text
            if self._history_text:
                display = self._history_text + " " + text
            else:
                display = text
        
        self.caption_display.setPlainText(display)
        
        # Scroll to bottom
        scrollbar = self.caption_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
        # Update status back to listening
        self._update_status_listening()
    
    def _update_status_listening(self):
        """Update status to show actively listening."""
        if self.is_listening:
            self.status_label.setText("● Listening...")
            self.status_label.setStyleSheet("""
                font-size: 16px;
                font-weight: bold;
                color: #4CAF50;
                padding: 10px;
            """)
    
    def _show_error(self, message: str):
        """Show an error message in the status."""
        self.status_label.setText(f"Error")
        self.status_label.setStyleSheet("""
            font-size: 16px;
            font-weight: bold;
            color: #f44336;
            padding: 10px;
        """)
        self.status_label.setToolTip(message)
        
        # Reset status after a delay
        QTimer.singleShot(3000, self._update_status_listening)
    
    def on_audio_error(self, error_message: str):
        """Handle audio recording errors."""
        self._show_error(error_message)
        print(f"Audio error: {error_message}")
    
    def on_recording_started(self):
        """Handle recording started event."""
        self.status_label.setText("Listening...")
        self.status_label.setStyleSheet("""
            font-size: 16px;
            font-weight: bold;
            color: #4CAF50;
            padding: 10px;
        """)
    
    def on_recording_stopped(self):
        """Handle recording stopped event."""
        self.status_label.setText("Ready")
        self.status_label.setStyleSheet("""
            font-size: 16px;
            font-weight: bold;
            color: #888;
            padding: 10px;
        """)
    
    def clear_captions(self):
        """Clear the caption display and server-side audio buffer."""
        self.caption_display.clear()
        self._live_text = ""
        self._history_text = ""
        if self.sio.connected:
            try:
                self.sio.emit('clear_buffer')
            except Exception:
                pass
    
    # ===== CRISIS MODE ALERT METHODS =====
    
    def _show_alert_overlay(self, alert_message: str):
        """
        Show the crisis mode alert overlay with flashing animation.
        Called via signal when backend detects an alert (e.g., fire alarm).
        """
        self.current_alert = alert_message
        
        # Update alert text based on the alert type
        alert_upper = alert_message.upper()
        if "FIRE" in alert_upper:
            self.alert_text_label.setText("DETECTED FIRE ALARM")
            self.alert_icon_label.setText("🔥")
        elif "SMOKE" in alert_upper:
            self.alert_text_label.setText("DETECTED SMOKE ALARM")
            self.alert_icon_label.setText("💨")
        elif "CARBON" in alert_upper or "CO" in alert_upper:
            self.alert_text_label.setText("DETECTED CO ALARM")
            self.alert_icon_label.setText("☠️")
        elif "SIREN" in alert_upper or "EMERGENCY" in alert_upper:
            self.alert_text_label.setText("DETECTED EMERGENCY SIREN")
            self.alert_icon_label.setText("🚨")
        elif "DOOR" in alert_upper:
            self.alert_text_label.setText("DETECTED DOORBELL / KNOCK")
            self.alert_icon_label.setText("🔔")
        else:
            self.alert_text_label.setText("DETECTED ALERT")
            self.alert_icon_label.setText("⚠️")
        
        # Hide the detail label since "DETECTED" is now in the main text
        self.alert_detail_label.setText("")
        self.alert_detail_label.setVisible(False)
        
        # Position overlay in center of widget
        self._position_alert_overlay()
        
        # Show overlay and start flashing
        self.alert_overlay.setVisible(True)
        self.alert_overlay.raise_()  # Bring to front
        self.alert_flash_state = False
        self.alert_flash_timer.start(self.ALERT_FLASH_INTERVAL_MS)
        
        # Trigger system beep for haptic feedback (runs in background thread)
        threading.Thread(target=self._play_alert_sound, daemon=True).start()
        
        print(f"[CRISIS MODE] Alert activated: {alert_message}")
    
    def _hide_alert_overlay(self):
        """Hide the crisis mode alert overlay."""
        self.alert_flash_timer.stop()
        self.alert_overlay.setVisible(False)
        self.current_alert = None
        self.alert_flash_state = False
        
        # Reset overlay to default red color
        self.alert_overlay.setStyleSheet("""
            QFrame#alertOverlay {
                background-color: #D32F2F;
                border: 4px solid #FFEB3B;
                border-radius: 15px;
            }
        """)
        print("[CRISIS MODE] Alert dismissed")
    
    def _toggle_alert_flash(self):
        """Toggle the alert overlay flash state (called by timer)."""
        self.alert_flash_state = not self.alert_flash_state
        
        if self.alert_flash_state:
            # Flash to bright yellow/orange
            self.alert_overlay.setStyleSheet("""
                QFrame#alertOverlay {
                    background-color: #FF6F00;
                    border: 4px solid #FFFFFF;
                    border-radius: 15px;
                }
            """)
        else:
            # Flash back to red
            self.alert_overlay.setStyleSheet("""
                QFrame#alertOverlay {
                    background-color: #D32F2F;
                    border: 4px solid #FFEB3B;
                    border-radius: 15px;
                }
            """)
    
    def _dismiss_alert(self):
        """Manually dismiss the alert overlay."""
        self._hide_alert_overlay()
    
    def _position_alert_overlay(self):
        """Position the alert overlay in the center of the widget."""
        # Calculate centered position
        overlay_width = min(500, self.width() - 40)
        overlay_height = min(300, self.height() - 100)
        x = (self.width() - overlay_width) // 2
        y = (self.height() - overlay_height) // 2
        
        self.alert_overlay.setGeometry(x, y, overlay_width, overlay_height)
    
    def _play_alert_sound(self):
        """Play system alert sounds for haptic/audio feedback (runs in thread)."""
        try:
            # Play attention-grabbing beep pattern
            for _ in range(3):
                winsound.Beep(2500, 200)  # High-pitched beep
                winsound.Beep(1500, 200)  # Lower beep
        except Exception as e:
            # winsound may not work on all systems
            print(f"[CRISIS MODE] Could not play alert sound: {e}")
    
    def resizeEvent(self, event):
        """Handle widget resize to reposition alert overlay."""
        super().resizeEvent(event)
        if self.alert_overlay.isVisible():
            self._position_alert_overlay()
    
    # ===== END CRISIS MODE METHODS =====
    
    def showEvent(self, event):
        """Called when widget is shown."""
        super().showEvent(event)
        # User starts recording manually
    
    def hideEvent(self, event):
        """Called when widget is hidden."""
        super().hideEvent(event)
        # Stop recording when leaving the view
        if self.is_listening:
            self.stop_listening()


class BackendCommunicator:
    """Handles communication with the backend server."""
    
    def __init__(self, base_url="http://127.0.0.1:5000"):
        self.base_url = base_url
        
    def check_status(self):
        """
        Check the backend status by calling the /status endpoint.
        
        Returns:
            tuple: (success: bool, message: str)
        """
        try:
            response = requests.get(
                f"{self.base_url}/status",
                timeout=2
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "ok":
                    return True, "Connected"
                else:
                    return False, "Unexpected response from backend"
            else:
                return False, f"Backend returned status {response.status_code}"
                
        except requests.exceptions.ConnectionError:
            return False, "Backend not running"
        except requests.exceptions.Timeout:
            return False, "Connection timeout"
        except Exception as e:
            return False, f"Error: {str(e)}"


class AtlasMainWindow(QMainWindow):
    """Main application window for Atlas."""
    
    def __init__(self):
        super().__init__()
        self.backend = BackendCommunicator()
        self.init_ui()
        self.check_backend_status()
        
        # Set up periodic backend status checking
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.check_backend_status)
        self.status_timer.start(3000)  # Check every 3 seconds
        
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Atlas")
        self.setMinimumSize(800, 600)
        
        # Create central widget with stacked layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Top bar with status indicator
        top_bar = QWidget()
        top_bar.setStyleSheet("background-color: #f5f5f5; border-bottom: 1px solid #ddd;")
        top_bar.setFixedHeight(40)
        top_bar_layout = QHBoxLayout(top_bar)
        top_bar_layout.setContentsMargins(10, 10, 10, 10)
        
        top_bar_layout.addStretch()
        
        # Status indicator
        self.status_indicator = StatusIndicator()
        top_bar_layout.addWidget(self.status_indicator)
        
        main_layout.addWidget(top_bar)
        
        # Stacked widget for different views
        self.stacked_widget = QStackedWidget()
        main_layout.addWidget(self.stacked_widget)
        
        # Create views
        self.mode_selection_view = ModeSelectionView()
        self.vision_mode_view = VisionModeWidget()
        self.hearing_mode_view = HearingModeWidget()
        
        # Add views to stacked widget
        self.stacked_widget.addWidget(self.mode_selection_view)
        self.stacked_widget.addWidget(self.vision_mode_view)
        self.stacked_widget.addWidget(self.hearing_mode_view)
        
        # Connect signals
        self.mode_selection_view.vision_button.clicked.connect(self.show_vision_mode)
        self.mode_selection_view.hearing_button.clicked.connect(self.show_hearing_mode)
        self.vision_mode_view.back_button.clicked.connect(self.show_mode_selection)
        self.hearing_mode_view.back_button.clicked.connect(self.show_mode_selection)
        
    def check_backend_status(self):
        """Check the backend status and update the indicator."""
        success, message = self.backend.check_status()
        
        if success:
            self.status_indicator.set_connected()
        else:
            self.status_indicator.set_disconnected(message)
    
    def show_vision_mode(self):
        """Switch to the Vision Assist Mode view."""
        self.stacked_widget.setCurrentWidget(self.vision_mode_view)
        
    def show_hearing_mode(self):
        """Switch to the Hearing Assist Mode view."""
        self.stacked_widget.setCurrentWidget(self.hearing_mode_view)
        
    def show_mode_selection(self):
        """Switch back to the Mode Selection view."""
        self.stacked_widget.setCurrentWidget(self.mode_selection_view)


def main():
    """Main entry point for the application."""
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle("Fusion")
    
    # Create and show main window
    window = AtlasMainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
