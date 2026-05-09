"""Threaded PyAudio capture for Hearing Mode, emitting WAV chunks for Whisper."""

import io
import wave
import threading
import pyaudio
from typing import Optional
from PyQt6.QtCore import QObject, pyqtSignal

class AudioRecorder(QObject):
    """Threaded PyAudio recorder that emits 16kHz mono WAV chunks via Qt signals."""
    
    # PyQt Signals for thread-safe communication
    audio_ready = pyqtSignal(bytes)       # Emits WAV audio data
    error_occurred = pyqtSignal(str)      # Emits error messages
    recording_started = pyqtSignal()      # Emitted when recording starts
    recording_stopped = pyqtSignal()      # Emitted when recording stops
    
    # Audio format constants optimized for Whisper
    SAMPLE_RATE = 16000       # 16kHz - optimal for speech recognition
    CHANNELS = 1              # Mono - Whisper expects mono
    SAMPLE_WIDTH = 2          # 16-bit audio (2 bytes per sample)
    FORMAT = pyaudio.paInt16
    CHUNK_SIZE = 1024         # Buffer size for reading audio
    
    def __init__(self, parent: Optional[QObject] = None):
        """
        Initialize the AudioRecorder.
        
        Args:
            parent: QObject for Qt memory management.
        """
        super().__init__(parent)
        
        self._pyaudio: Optional[pyaudio.PyAudio] = None
        self._stream: Optional[pyaudio.Stream] = None
        self._recording_thread: Optional[threading.Thread] = None
        self._is_recording: bool = False
        self._stop_event: threading.Event = threading.Event()
        self._chunk_duration: float = 0.5  # 0.5 seconds per chunk for live streaming
        
    @property
    def is_recording(self) -> bool:
        """Check if the recorder is currently active."""
        return self._is_recording
    
    def start_listening(self, chunk_duration: float = 0.5) -> bool:
        """
        Start recording audio from the microphone in a background thread.
        Emits audio chunks via audio_ready signal.
        Args:
            chunk_duration: Duration of each audio chunk in seconds (default: 0.5).
        
        Returns:
            bool: True if recording started successfully, False otherwise.
        """

        if self._is_recording:
            self.error_occurred.emit("Recording is already in progress.")
            return False
        
        self._chunk_duration = max(0.3, min(chunk_duration, 30.0))  # Clamp to 0.3-30 sec
        self._stop_event.clear()
        
        # Start recording in a background thread
        self._recording_thread = threading.Thread(
            target=self._recording_loop,
            name="AtlasAudioRecorder",
            daemon=True
        )
        self._recording_thread.start()
        
        return True
    
    def stop_listening(self) -> None:
        """
        Stop the audio recording and clean up resources.
        """
        if not self._is_recording:
            return
        
        # Signal the thread to stop
        self._stop_event.set()
        
        # Wait for thread to finish (with timeout)
        if self._recording_thread and self._recording_thread.is_alive():
            self._recording_thread.join(timeout=2.0)
        
        self._recording_thread = None
    
    def _recording_loop(self) -> None:
        """
        Main recording loop (background thread).
        Records audio chunks and emits them via signals.
        """
        try:
            # Initialize PyAudio
            self._pyaudio = pyaudio.PyAudio()
            
            # Open audio stream from default microphone
            self._stream = self._pyaudio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.SAMPLE_RATE,
                input=True,
                frames_per_buffer=self.CHUNK_SIZE
            )
            
            self._is_recording = True
            self.recording_started.emit()
            
            # Calculate number of chunks needed for desired duration
            chunks_per_segment = int(
                (self.SAMPLE_RATE * self._chunk_duration) / self.CHUNK_SIZE
            )
            
            # Continuous recording loop
            while not self._stop_event.is_set():
                audio_frames = []
                
                # Record one chunk of audio
                for _ in range(chunks_per_segment):
                    if self._stop_event.is_set():
                        break
                    
                    try:
                        data = self._stream.read(
                            self.CHUNK_SIZE,
                            exception_on_overflow=False
                        )
                        audio_frames.append(data)
                    except Exception as e:
                        # Handle buffer overflow or other read errors gracefully
                        if not self._stop_event.is_set():
                            print(f"Audio read warning: {e}")
                        continue
                
                # If we have audio data and weren't stopped mid-recording
                if audio_frames and not self._stop_event.is_set():
                    # Convert to WAV format
                    wav_data = self._frames_to_wav(audio_frames)
                    if wav_data:
                        # Emit the audio data (thread-safe via signal)
                        self.audio_ready.emit(wav_data)
        
        except Exception as e:
            error_msg = f"Recording error: {str(e)}"
            print(error_msg)
            self.error_occurred.emit(error_msg)
        
        finally:
            # Clean up resources
            self._cleanup_stream()
            self._is_recording = False
            self.recording_stopped.emit()
    
    def _frames_to_wav(self, frames: list) -> bytes:
        """
        Convert raw audio frames to WAV format.
        
        Args:
            frames: List of raw audio data chunks.
        
        Returns:
            bytes: WAV-formatted audio data, or empty bytes on error.
        """
        try:
            # Create an in-memory WAV file
            wav_buffer = io.BytesIO()
            
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(self.CHANNELS)
                wav_file.setsampwidth(self.SAMPLE_WIDTH)
                wav_file.setframerate(self.SAMPLE_RATE)
                wav_file.writeframes(b''.join(frames))
            
            wav_buffer.seek(0)
            return wav_buffer.read()
        
        except Exception as e:
            print(f"Error creating WAV data: {e}")
            return b''
    
    def _cleanup_stream(self) -> None:
        """Clean up PyAudio stream and resources."""
        try:
            if self._stream:
                self._stream.stop_stream()
                self._stream.close()
                self._stream = None
        except Exception as e:
            print(f"Error closing stream: {e}")
        
        try:
            if self._pyaudio:
                self._pyaudio.terminate()
                self._pyaudio = None
        except Exception as e:
            print(f"Error terminating PyAudio: {e}")
    
    def get_input_devices(self) -> list:
        """
        Get a list of available audio input devices.
        
        Returns:
            list: List of dicts with 'index', 'name', and 'channels' keys.
        """
        devices = []
        pa = pyaudio.PyAudio()
        for i in range(pa.get_device_count()):
            device_info = pa.get_device_info_by_index(i)
            if device_info.get('maxInputChannels', 0) > 0:
                devices.append({
                    'index': i,
                    'name': device_info.get('name', 'Unknown'),
                    'channels': device_info.get('maxInputChannels', 0)
                })
        pa.terminate()
        return devices


class ContinuousAudioRecorder(AudioRecorder):
    """AudioRecorder variant that uses chunk overlap to avoid clipped words."""
    
    def __init__(self, parent: Optional[QObject] = None, overlap_duration: float = 0.5):
        """
        Initialize the ContinuousAudioRecorder.
        
        Args:
            parent: QObject.
            overlap_duration: Overlap between chunks in seconds (default: 0.5).
        """
        super().__init__(parent)
        self._overlap_duration = overlap_duration
        self._overlap_buffer: bytes = b''
    
    def _recording_loop(self) -> None:
        """
        Recording loop with overlap support for seamless transcription.
        """
        try:
            self._pyaudio = pyaudio.PyAudio()
            
            self._stream = self._pyaudio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.SAMPLE_RATE,
                input=True,
                frames_per_buffer=self.CHUNK_SIZE
            )
            
            self._is_recording = True
            self.recording_started.emit()
            
            # Calculate chunks for main duration and overlap
            chunks_per_segment = int(
                (self.SAMPLE_RATE * self._chunk_duration) / self.CHUNK_SIZE
            )
            overlap_chunks = int(
                (self.SAMPLE_RATE * self._overlap_duration) / self.CHUNK_SIZE
            )
            
            overlap_frames = []
            
            while not self._stop_event.is_set():
                audio_frames = list(overlap_frames)  # Start with overlap from previous
                overlap_frames = []
                
                # Record new audio
                for i in range(chunks_per_segment):
                    if self._stop_event.is_set():
                        break
                    
                    try:
                        data = self._stream.read(
                            self.CHUNK_SIZE,
                            exception_on_overflow=False
                        )
                        audio_frames.append(data)
                        
                        # Keep last N chunks for overlap
                        if i >= chunks_per_segment - overlap_chunks:
                            overlap_frames.append(data)
                    
                    except Exception as e:
                        if not self._stop_event.is_set():
                            print(f"Audio read warning: {e}")
                        continue
                
                if audio_frames and not self._stop_event.is_set():
                    wav_data = self._frames_to_wav(audio_frames)
                    if wav_data:
                        self.audio_ready.emit(wav_data)
        
        except Exception as e:
            error_msg = f"Recording error: {str(e)}"
            print(error_msg)
            self.error_occurred.emit(error_msg)
        
        finally:
            self._cleanup_stream()
            self._is_recording = False
            self.recording_stopped.emit()