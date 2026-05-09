from PyQt6.QtWidgets import QLabel
from PyQt6.QtGui import QPainter, QPen, QFont, QColor
from PyQt6.QtCore import Qt


class OverlayLabel(QLabel):
    """Custom QLabel widget that draws ML detection data as overlay on video frames."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.detection_data = None
        self.original_frame_size = None  # Store the size of frame sent to backend
        
    def update_data(self, new_data, frame_size=None):
        """
        Update the detection data and trigger a repaint.
        Args:
            new_data: Detection results from backend
            frame_size: (width, height) tuple of the original frame sent to backend
        """
        self.detection_data = new_data
        if frame_size:
            self.original_frame_size = frame_size
        self.update()  # Trigger repaint
    
    def paintEvent(self, event):
        """
        Override paintEvent to draw video frame and detection overlays.
        Called automatically when update() is triggered.
        """
        # Draw the parent's content (the video frame)
        super().paintEvent(event)
        
        # Skip overlay if no detection data or frame size
        if not self.detection_data or not self.detection_data.get('objects'):
            return
        if not self.original_frame_size:
            return
            
        # Get the current pixmap to determine actual displayed size
        pixmap = self.pixmap()
        if not pixmap or pixmap.isNull():
            return
            
        # Calculate coordinate transformation
        # Original frame size (what backend analyzed)
        orig_width, orig_height = self.original_frame_size
        
        # Displayed pixmap size (scaled with KeepAspectRatio)
        display_width = pixmap.width()
        display_height = pixmap.height()
        
        # Label size (container)
        label_width = self.width()
        label_height = self.height()
        
        # Calculate scale factors
        scale_x = display_width / orig_width
        scale_y = display_height / orig_height
        
        # Calculate offset (centering due to KeepAspectRatio)
        offset_x = (label_width - display_width) / 2
        offset_y = (label_height - display_height) / 2
        
        # Create painter for drawing overlays
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Set up pen for bounding boxes
        pen = QPen(QColor(0, 255, 0))  # Green color
        pen.setWidth(3)
        painter.setPen(pen)
        
        # Set up font for labels
        font = QFont('Arial', 12, QFont.Weight.Bold)
        painter.setFont(font)
        
        # Draw each detected object
        for obj in self.detection_data['objects']:
            # Extract bounding box coordinates (in original frame space)
            box = obj.get('box', [])
            if len(box) != 4:
                continue
            
            x_min, y_min, x_max, y_max = box
            
            # Transform coordinates to displayed space
            x_min_display = int(x_min * scale_x + offset_x)
            y_min_display = int(y_min * scale_y + offset_y)
            x_max_display = int(x_max * scale_x + offset_x)
            y_max_display = int(y_max * scale_y + offset_y)
            
            # Calculate width and height in display space
            width_display = x_max_display - x_min_display
            height_display = y_max_display - y_min_display
            
            # Draw bounding box rectangle
            painter.drawRect(x_min_display, y_min_display, width_display, height_display)
            
            # Prepare label text with confidence
            label = obj.get('label', 'Unknown')
            confidence = obj.get('confidence', 0.0)
            text = f"{label} {confidence:.0%}"
            
            # Draw text background for readability
            text_rect = painter.fontMetrics().boundingRect(text)
            text_bg_rect = text_rect.adjusted(-4, -2, 4, 2)
            text_bg_rect.moveTopLeft(painter.fontMetrics().boundingRect(x_min_display, y_min_display, 0, 0, 0, text).topLeft())
            text_bg_rect.translate(x_min_display, y_min_display - text_bg_rect.height())
            
            painter.fillRect(text_bg_rect, QColor(0, 0, 0, 180))  # Semi-transparent black
            
            # Draw label text above the box (in display coordinates)
            painter.setPen(QColor(255, 255, 255))  # White text
            painter.drawText(x_min_display, y_min_display - 5, text)
            
            # Reset pen for next box
            painter.setPen(pen)
        
        painter.end()
