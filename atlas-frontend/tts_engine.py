from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot
import pyttsx3
import queue
import threading


class TTSWorker(QObject):
    """TTS worker"""
    
    # Signal emitted when speech starts
    started = pyqtSignal()
    # Signal emitted when speech finishes
    finished = pyqtSignal()
    # Signal emitted on error
    error = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.engine = None
        self.is_speaking = False
        self.should_stop = False
        self._lock = threading.Lock()
        
    def initialize_engine(self):
        """Initialize the pyttsx3 engine on the worker thread."""
        try:
            self.engine = pyttsx3.init()
            # Set default properties
            self.engine.setProperty('rate', 150)  # Speed of speech
            self.engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)
        except Exception as e:
            self.error.emit(f"Failed to initialize TTS engine: {e}")
    
    @pyqtSlot(str)
    def speak_text(self, text):
        """Speak the given text. Interrupts current speech if any."""
        if not self.engine:
            self.initialize_engine()
        
        if not text or not text.strip():
            return
        
        with self._lock:
            # Stop any ongoing speech
            if self.is_speaking:
                try:
                    self.engine.stop()
                    # Wait a bit for the engine to actually stop
                    import time
                    time.sleep(0.1)
                except:
                    pass
            
            self.is_speaking = True
            self.should_stop = False
        
        try:
            self.started.emit()
            # Clear any pending text
            self.engine.stop()
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            # Ignore "run loop already started" errors as they're expected during interruption
            if "run loop already started" not in str(e).lower():
                self.error.emit(f"TTS error: {e}")
        finally:
            with self._lock:
                self.is_speaking = False
            self.finished.emit()
    
    @pyqtSlot()
    def stop_speaking(self):
        """Stop current speech immediately."""
        with self._lock:
            if self.is_speaking and self.engine:
                try:
                    self.engine.stop()
                    self.should_stop = True
                except:
                    pass


class TTSEngine(QObject):
    """
    Non-blocking TTS engine controller.
    Manages threading and provides simple interface for the UI.
    """
    
    # Signal to request speech
    request_speak = pyqtSignal(str)
    # Signal to request stop
    request_stop = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        
        # Create worker thread
        self.thread = QThread()
        self.worker = TTSWorker()
        
        # Move worker to thread
        self.worker.moveToThread(self.thread)
        
        # Connect signals
        self.request_speak.connect(self.worker.speak_text)
        self.request_stop.connect(self.worker.stop_speaking)
        
        # Forward worker signals
        self.worker.started.connect(self._on_started)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        
        # Initialize engine when thread starts
        self.thread.started.connect(self.worker.initialize_engine)
        
        # Start the thread
        self.thread.start()
        
        self._is_speaking = False
    
    def say(self, text):
        """
        Speak the given text asynchronously.
        Interrupts any current speech.
        """
        if text and text.strip():
            self.request_speak.emit(text)
    
    def stop(self):
        """Stop current speech immediately."""
        self.request_stop.emit()
    
    def is_speaking(self):
        """Check if currently speaking."""
        return self._is_speaking
    
    def shutdown(self):
        """Shutdown the TTS engine and thread."""
        self.stop()
        self.thread.quit()
        self.thread.wait()
    
    def _on_started(self):
        self._is_speaking = True
    
    def _on_finished(self):
        self._is_speaking = False
    
    def _on_error(self, error_msg):
        print(f"TTS Error: {error_msg}")
        self._is_speaking = False
