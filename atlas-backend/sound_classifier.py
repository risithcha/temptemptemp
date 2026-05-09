"""
Environmental sound classification for Atlas.
Detects sirens, alarms, and similar high-priority sounds using YAMNet and FFT.
"""

import numpy as np
import io
import wave
import threading
from typing import Optional, Dict, List, Tuple
import os

# Sound classification result types
ALERT_TYPES = {
    "FIRE_ALARM": "FIRE ALARM DETECTED",
    "SMOKE_ALARM": "SMOKE ALARM DETECTED", 
    "SIREN": "EMERGENCY SIREN DETECTED",
    "CAR_ALARM": "CAR ALARM DETECTED",
    "HORN": "HORN/ALERT DETECTED",
    "GLASS_BREAK": "GLASS BREAKING DETECTED",
    "DOORBELL": "DOORBELL DETECTED",
    "SCREAM": "SCREAM/DISTRESS DETECTED",
}

# YAMNet class names that map to alerts (from yamnet_class_map.csv)
# These are the display_name values from AudioSet ontology
YAMNET_ALERT_CLASSES = {
    # Fire/Smoke alarms
    "Fire alarm": "FIRE_ALARM",
    "Smoke detector, smoke alarm": "SMOKE_ALARM",
    "Alarm": "FIRE_ALARM",
    "Alarm clock": None,  # Exclude - not an emergency
    
    # Sirens
    "Siren": "SIREN",
    "Civil defense siren": "SIREN",
    "Ambulance (siren)": "SIREN",
    "Fire engine, fire truck (siren)": "SIREN",
    "Police car (siren)": "SIREN",
    
    # Vehicle alerts
    "Car alarm": "CAR_ALARM",
    "Vehicle horn, car horn, honking": "HORN",
    "Truck horn, air horn": "HORN",
    "Bicycle bell": None,  # Less urgent
    
    # Other alerts
    "Bell": "DOORBELL",
    "Doorbell": "DOORBELL",
    "Buzzer": "FIRE_ALARM",
    "Glass": None,  # Too generic
    "Shatter": "GLASS_BREAK",
    "Breaking": "GLASS_BREAK",
    
    # Human distress
    "Screaming": "SCREAM",
    "Shout": "SCREAM",
    "Crying, sobbing": None,  # Could be false positive
}

# Frequency ranges for common alarm types (in Hz)
# Tuned to avoid false positives from speech
# Speech fundamentals: Male ~85-180Hz, Female ~165-255Hz
# Speech formants (vowels): F1=300-900Hz, F2=850-2500Hz
ALARM_FREQUENCIES = {
    "smoke_alarm": (2800, 4500),      # Smoke detectors: 3000-4000Hz (very high pitched)
    "fire_alarm": (2400, 4000),       # Fire alarms: 2500-4000Hz (high pitched)
    "siren_high": (1800, 3500),       # High siren sweep (above speech formants)
    "siren_mid": (1000, 1800),        # Mid siren sweep (raised from 500 to avoid speech)
    "car_alarm": (1200, 2500),        # Car alarms: raised minimum to avoid speech
}

# Spectral characteristics to distinguish alarms from speech
ALARM_DETECTION_PARAMS = {
    "min_energy_ratio": 0.45,          # Minimum 45% of energy in alarm band
    "min_peak_prominence": 4.0,        # Peak must be 4x the mean spectrum
    "min_spectral_centroid": 1200,     # Alarms have high spectral centroid (Hz)
    "max_spectral_flatness": 0.3,      # Alarms are tonal (low flatness), speech is noisy (high)
    "min_pulsing_cv": 0.35,            # Coefficient of variation for pulsing detection
}


class SoundClassifier:
    """
    Environmental Sound Classifier for detecting emergency sounds.
    Uses YAMNet first with a strict timeout; if it doesn't return in time or no alert is detected, falls back to fast FFT.
    """
    
    def __init__(self, use_yamnet: bool = True):
        """
        Initialize the sound classifier.
        
        Args:
            use_yamnet: Whether to try loading YAMNet model
        """
        self.yamnet_model = None
        self.yamnet_classes = None
        self.yamnet_available = False
        
        if use_yamnet:
            self._try_load_yamnet()
    
    def _try_load_yamnet(self):
        """Attempt to load YAMNet model from TensorFlow Hub."""
        try:
            import tensorflow_hub as hub
            import tensorflow as tf
            import csv
            self.yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
            # Load class names from the model's asset
            class_map_path = self.yamnet_model.class_map_path().numpy().decode('utf-8')
            with tf.io.gfile.GFile(class_map_path) as f:
                reader = csv.DictReader(f)
                self.yamnet_classes = [row['display_name'] for row in reader]
            self.yamnet_available = True
            print("[SoundClassifier] YAMNet loaded successfully")
        except Exception as e:
            print(f"[SoundClassifier] YAMNet failed to load, falling back to FFT-only: {e}")
            self.yamnet_model = None
            self.yamnet_classes = None
            self.yamnet_available = False

    def _parse_wav_bytes(self, audio_data: bytes) -> Tuple[np.ndarray, int]:
        """
        Parse WAV audio bytes into numpy array and sample rate.
        
        Args:
            audio_data: Raw WAV file bytes
            
        Returns:
            Tuple of (waveform as float32 array normalized to [-1, 1], sample_rate)
        """
        with io.BytesIO(audio_data) as audio_buffer:
            with wave.open(audio_buffer, 'rb') as wav_file:
                sample_rate = wav_file.getframerate()
                n_channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                n_frames = wav_file.getnframes()
                
                # Read raw audio data
                raw_data = wav_file.readframes(n_frames)
                
                # Convert to numpy array based on sample width
                if sample_width == 1:
                    dtype = np.uint8
                    max_val = 255.0
                    offset = 128
                elif sample_width == 2:
                    dtype = np.int16
                    max_val = 32768.0
                    offset = 0
                elif sample_width == 4:
                    dtype = np.int32
                    max_val = 2147483648.0
                    offset = 0
                else:
                    raise ValueError(f"Unsupported sample width: {sample_width}")
                
                audio = np.frombuffer(raw_data, dtype=dtype)
                
                # Convert to float and normalize to [-1, 1]
                audio = (audio.astype(np.float32) - offset) / max_val
                
                # Convert stereo to mono by averaging channels
                if n_channels == 2:
                    audio = audio.reshape(-1, 2).mean(axis=1)
                elif n_channels > 2:
                    audio = audio.reshape(-1, n_channels).mean(axis=1)
                
                return audio, sample_rate
    
    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Simple resampling using linear interpolation."""
        if orig_sr == target_sr:
            return audio
        
        duration = len(audio) / orig_sr
        target_length = int(duration * target_sr)
        
        # Use numpy's interp for simple resampling
        x_orig = np.linspace(0, duration, len(audio))
        x_new = np.linspace(0, duration, target_length)
        return np.interp(x_new, x_orig, audio).astype(np.float32)
    
    def _analyze_fft(self, audio: np.ndarray, sample_rate: int) -> Dict:
        """
        Analyze audio with FFT and spectral features to detect alarms and suppress speech.
        """
        # Ensure we have enough samples for meaningful analysis
        if len(audio) < sample_rate * 0.1:  # At least 100ms
            return {"detected": False, "reason": "Audio too short"}
        
        # Apply FFT
        n_fft = min(4096, len(audio))
        
        # Analyze in overlapping windows to catch pulsing patterns
        hop_length = n_fft // 2
        n_windows = max(1, (len(audio) - n_fft) // hop_length + 1)
        
        freq_bins = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
        
        detections = []
        energy_history = []
        centroid_history = []
        peak_freq_history = []  # Track dominant frequency across windows for sweep detection
        
        # Get detection parameters
        params = ALARM_DETECTION_PARAMS
        
        for i in range(min(n_windows, 10)):  # Analyze up to 10 windows
            start = i * hop_length
            end = start + n_fft
            if end > len(audio):
                break
                
            window = audio[start:end] * np.hanning(n_fft)
            spectrum = np.abs(np.fft.rfft(window))
            
            # Calculate total energy
            total_energy = np.sum(spectrum ** 2)
            energy_history.append(total_energy)
            
            # Skip near-silent windows
            if total_energy < 1e-8:
                continue
            
            # Calculate spectral centroid (weighted average frequency)
            # Alarms/sirens: typically 1500-4000 Hz, Speech: typically 500-1500 Hz
            spectral_centroid = np.sum(freq_bins * spectrum) / (np.sum(spectrum) + 1e-10)
            centroid_history.append(spectral_centroid)
            
            # Calculate spectral flatness (0=pure tone, 1=white noise)
            # Alarms are tonal (low flatness), speech is noisier (higher flatness)
            log_spectrum = np.log(spectrum + 1e-10)
            geometric_mean = np.exp(np.mean(log_spectrum))
            arithmetic_mean = np.mean(spectrum)
            spectral_flatness = geometric_mean / (arithmetic_mean + 1e-10)
            
            # Find overall peak frequency for sweep detection
            overall_peak_idx = np.argmax(spectrum)
            overall_peak_freq = freq_bins[overall_peak_idx]
            peak_freq_history.append(overall_peak_freq)
            
            # Only consider audio with alarm-like spectral characteristics
            # Speech typically has centroid < 1200 Hz and flatness > 0.3
            if spectral_centroid < params["min_spectral_centroid"]:
                continue  # Too speech-like
            
            if spectral_flatness > params["max_spectral_flatness"]:
                continue  # Too noisy/speech-like
            
            # Check each alarm frequency range
            for alarm_type, (low_freq, high_freq) in ALARM_FREQUENCIES.items():
                # Find frequency bin indices
                low_idx = np.searchsorted(freq_bins, low_freq)
                high_idx = np.searchsorted(freq_bins, high_freq)
                
                if high_idx <= low_idx:
                    continue
                
                # Calculate energy in this band
                band_energy = np.sum(spectrum[low_idx:high_idx] ** 2)
                band_ratio = band_energy / (total_energy + 1e-10)
                
                # Find peak frequency in band
                band_spectrum = spectrum[low_idx:high_idx]
                if len(band_spectrum) > 0:
                    peak_idx = np.argmax(band_spectrum)
                    peak_freq = freq_bins[low_idx + peak_idx]
                    peak_magnitude = band_spectrum[peak_idx]
                    
                    # Check if this band is dominant with stricter thresholds
                    if (band_ratio > params["min_energy_ratio"] and 
                        peak_magnitude > np.mean(spectrum) * params["min_peak_prominence"]):
                        detections.append({
                            "type": alarm_type,
                            "peak_freq": float(peak_freq),
                            "energy_ratio": float(band_ratio),
                            "spectral_centroid": float(spectral_centroid),
                            "spectral_flatness": float(spectral_flatness),
                            "window": i
                        })
        
        # Check for pulsing pattern (characteristic of alarms)
        is_pulsing = False
        if len(energy_history) >= 3:
            energy_std = np.std(energy_history)
            energy_mean = np.mean(energy_history)
            if energy_mean > 0:
                # High coefficient of variation suggests pulsing
                cv = energy_std / energy_mean
                is_pulsing = cv > params["min_pulsing_cv"]
        
        # Check for frequency sweep pattern (characteristic of sirens)
        is_sweeping = False
        if len(peak_freq_history) >= 3:
            freq_range = max(peak_freq_history) - min(peak_freq_history)
            # Sirens typically sweep 500-2000 Hz range
            is_sweeping = freq_range > 300  # At least 300 Hz variation
        
        # Check if spectral centroid is consistently high (alarm-like)
        avg_centroid = np.mean(centroid_history) if centroid_history else 0
        
        # Aggregate detections
        if detections:
            # Count detection types
            type_counts = {}
            for d in detections:
                t = d["type"]
                type_counts[t] = type_counts.get(t, 0) + 1
            
            # Get most frequent detection
            best_type = max(type_counts, key=type_counts.get)
            confidence = type_counts[best_type] / len(detections)
            
            # Boost confidence for sweeping sirens or pulsing alarms
            if is_sweeping and "siren" in best_type:
                confidence = min(1.0, confidence * 1.2)
            if is_pulsing and ("alarm" in best_type or "fire" in best_type):
                confidence = min(1.0, confidence * 1.2)
            
            # Map to alert type
            if "smoke" in best_type:
                alert_key = "SMOKE_ALARM"
            elif "fire" in best_type:
                alert_key = "FIRE_ALARM" 
            elif "siren" in best_type:
                alert_key = "SIREN"
            elif "car" in best_type:
                alert_key = "CAR_ALARM"
            else:
                alert_key = "FIRE_ALARM"  # Default to fire alarm for safety
            
            return {
                "detected": True,
                "method": "fft",
                "alert_type": alert_key,
                "alert_message": ALERT_TYPES.get(alert_key, "ALERT DETECTED"),
                "confidence": float(confidence),
                "is_pulsing": bool(is_pulsing),
                "is_sweeping": bool(is_sweeping),
                "avg_spectral_centroid": float(avg_centroid),
                "detections": detections[:5],  # First 5 detections
            }
        
        return {"detected": False, "method": "fft", "avg_spectral_centroid": float(avg_centroid)}
    
    def _analyze_yamnet(self, audio: np.ndarray, sample_rate: int) -> Dict:
        """
        Analyze audio with YAMNet (16 kHz mono) to detect alert classes.
        """
        if not self.yamnet_available:
            return {"detected": False, "reason": "YAMNet not available"}
        
        try:
            import tensorflow as tf
            
            # YAMNet expects 16kHz mono audio
            if sample_rate != 16000:
                audio = self._resample(audio, sample_rate, 16000)
            
            # Run inference
            scores, embeddings, spectrogram = self.yamnet_model(audio)
            scores = scores.numpy()
            
            # Average scores across time frames
            mean_scores = np.mean(scores, axis=0)
            
            # Get top predictions
            top_indices = np.argsort(mean_scores)[::-1][:20]
            
            # Check if any alert classes are in top predictions
            for idx in top_indices:
                class_name = self.yamnet_classes[idx]
                score = mean_scores[idx]
                
                # Only consider confident predictions
                if score < 0.1:
                    break
                
                # Check if this class maps to an alert
                if class_name in YAMNET_ALERT_CLASSES:
                    alert_key = YAMNET_ALERT_CLASSES[class_name]
                    if alert_key is not None:
                        return {
                            "detected": True,
                            "method": "yamnet",
                            "alert_type": alert_key,
                            "alert_message": ALERT_TYPES.get(alert_key, "ALERT DETECTED"),
                            "confidence": float(score),
                            "class_name": class_name,
                            "top_classes": [
                                {"class": self.yamnet_classes[i], "score": float(mean_scores[i])}
                                for i in top_indices[:5]
                            ]
                        }
            
            # Return top classes even if no alert detected (for debugging)
            return {
                "detected": False,
                "method": "yamnet",
                "top_classes": [
                    {"class": self.yamnet_classes[i], "score": float(mean_scores[i])}
                    for i in top_indices[:5]
                ]
            }
            
        except Exception as e:
            print(f"[SoundClassifier] YAMNet inference error: {e}")
            return {"detected": False, "method": "yamnet", "error": str(e)}
    
    def classify(self, audio_data: bytes) -> Dict:
        """
        Detect emergency sounds. Runs YAMNet first with a short timeout, then falls back to FFT.
        """
        try:
            # Parse WAV audio
            audio, sample_rate = self._parse_wav_bytes(audio_data)
            
            if len(audio) < 100:
                return {
                    "alert": None,
                    "alert_type": None,
                    "details": {"error": "Audio too short"}
                }
            
            # First: Try YAMNet if available (AI-first)
            if self.yamnet_available:
                try:
                    yamnet_result = self._analyze_yamnet(audio, sample_rate)
                except Exception as e:
                    yamnet_result = {"detected": False, "method": "yamnet", "error": str(e)}

                if yamnet_result.get("detected"):
                    return {
                        "alert": yamnet_result.get("alert_message"),
                        "alert_type": yamnet_result.get("alert_type"),
                        "confidence": yamnet_result.get("confidence"),
                        "method": "yamnet",
                        "details": yamnet_result
                    }
            else:
                yamnet_result = {"detected": False, "method": "yamnet", "available": False}

            # Second: Fall back to FFT-based detection (fast)
            fft_result = self._analyze_fft(audio, sample_rate)

            if fft_result.get("detected") and fft_result.get("confidence", 0) > 0.5:
                return {
                    "alert": fft_result.get("alert_message"),
                    "alert_type": fft_result.get("alert_type"),
                    "confidence": fft_result.get("confidence"),
                    "method": "fft",
                    "details": {
                        "fft_result": fft_result,
                        "yamnet_result": yamnet_result if self.yamnet_available else None
                    }
                }
            
            # No alert detected
            return {
                "alert": None,
                "alert_type": None,
                "details": {
                    "fft_result": fft_result,
                    "yamnet_result": yamnet_result if self.yamnet_available else None
                }
            }
            
        except Exception as e:
            print(f"[SoundClassifier] Classification error: {e}")
            return {
                "alert": None,
                "alert_type": None,
                "error": str(e)
            }


# Global instance (lazy initialization)
_classifier_instance: Optional[SoundClassifier] = None
_classifier_lock = threading.Lock()


def get_classifier() -> SoundClassifier:
    """Get or create the global SoundClassifier instance (thread-safe)."""
    global _classifier_instance
    if _classifier_instance is None:
        with _classifier_lock:
            if _classifier_instance is None:
                # Try to use YAMNet, but fall back to FFT-only if not available
                _classifier_instance = SoundClassifier(use_yamnet=True)
    return _classifier_instance


def classify_audio(audio_data: bytes) -> Dict:
    """
    Convenience function to classify audio data.
    
    Args:
        audio_data: Raw WAV file bytes
        
    Returns:
        Classification result dict with 'alert' key
    """
    return get_classifier().classify(audio_data)
