from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit as ws_emit
from vision_processor import detect_objects, preload_models as preload_vision
from sound_classifier import classify_audio, get_classifier
import tempfile
import os
import io
import wave
import base64
import threading
import numpy as np
from faster_whisper import WhisperModel
from concurrent.futures import ThreadPoolExecutor
import logging

# Configure logging for faster-whisper
logging.basicConfig()
logging.getLogger("faster_whisper").setLevel(logging.WARNING)

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# SocketIO for real-time WebSocket streaming (threading mode for compatibility)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Initialize Whisper model - 'tiny' is ~5-10x faster than 'base' on CPU
print("Loading Whisper model (tiny)...")
whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")
print("Whisper model loaded")

# Latest sound classification result (updated in background)
_latest_classification = {"alert": None, "alert_type": None}
_classification_lock = threading.Lock()

# Eagerly pre-load all models at startup so first requests are fast
print("Pre-loading vision models...")
preload_vision()
print("Vision models loaded")

print("Pre-loading sound classifier (YAMNet + TensorFlow)...")
get_classifier()
print("Sound classifier loaded")

# Initialize thread pool for parallel processing
executor = ThreadPoolExecutor(max_workers=2)

# ── WebSocket streaming state ──────────────────────────────────────────
# Per-client raw PCM buffer and transcription lock
_client_buffers = {}       # sid -> bytearray (raw 16-bit PCM)
_client_transcribing = {}  # sid -> bool
_client_buf_age = {}       # sid -> float (time first byte entered current buffer)
_ws_lock = threading.Lock()

MAX_BUFFER_SECONDS = 5     # finalize after this many seconds of audio
SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2           # 16-bit


def _extract_pcm_from_wav(wav_bytes: bytes) -> bytes:
    """Extract raw PCM samples from WAV container."""
    try:
        with wave.open(io.BytesIO(wav_bytes), 'rb') as wf:
            return wf.readframes(wf.getnframes())
    except Exception:
        return b''


# ── WebSocket events ───────────────────────────────────────────────────
@socketio.on('connect')
def _ws_connect():
    sid = request.sid
    with _ws_lock:
        _client_buffers[sid] = bytearray()
        _client_transcribing[sid] = False
        _client_buf_age[sid] = 0.0
    print(f"[WS] Client connected: {sid}")


@socketio.on('disconnect')
def _ws_disconnect():
    sid = request.sid
    with _ws_lock:
        _client_buffers.pop(sid, None)
        _client_transcribing.pop(sid, None)
        _client_buf_age.pop(sid, None)
    print(f"[WS] Client disconnected: {sid}")


@socketio.on('clear_buffer')
def _ws_clear_buffer():
    sid = request.sid
    with _ws_lock:
        _client_buffers[sid] = bytearray()
        _client_buf_age[sid] = 0.0


@socketio.on('audio_chunk')
def _ws_audio_chunk(data):
    """Receive a small WAV chunk, append to buffer, transcribe & emit.
    
    Emits 'transcription' with:
      - final=False  → interim preview (updates the live line in the UI)
      - final=True   → committed text (appended permanently), buffer is cleared
    """
    import time as _time
    sid = request.sid
    try:
        audio_bytes = base64.b64decode(data.get('audio', ''))
        pcm = _extract_pcm_from_wav(audio_bytes)
        if not pcm:
            return

        max_bytes = MAX_BUFFER_SECONDS * SAMPLE_RATE * SAMPLE_WIDTH
        now = _time.monotonic()

        with _ws_lock:
            buf = _client_buffers.get(sid)
            if buf is None:
                return
            if len(buf) == 0:
                _client_buf_age[sid] = now
            buf.extend(pcm)

            buf_duration = len(buf) / (SAMPLE_RATE * SAMPLE_WIDTH)
            is_final = buf_duration >= MAX_BUFFER_SECONDS

            # If already transcribing, skip (we'll catch up on next chunk)
            if _client_transcribing.get(sid, False):
                return
            _client_transcribing[sid] = True
            snapshot = bytes(buf)

            # If final, clear the buffer now so new audio goes into a fresh buffer
            if is_final:
                buf.clear()
                _client_buf_age[sid] = 0.0

        def _transcribe_and_emit():
            try:
                pcm_f32 = np.frombuffer(snapshot, dtype=np.int16).astype(np.float32) / 32768.0
                segments, info = whisper_model.transcribe(
                    pcm_f32,
                    beam_size=1,
                    language="en",
                    condition_on_previous_text=False,
                )
                text = " ".join(seg.text.strip() for seg in segments).strip()

                socketio.emit('transcription', {
                    'text': text,
                    'final': is_final,
                    'language': info.language,
                    'language_probability': info.language_probability,
                }, room=sid)

                # Background classification on the raw WAV chunk
                def _bg_classify():
                    global _latest_classification
                    try:
                        result = classify_audio(audio_bytes)
                        with _classification_lock:
                            _latest_classification = result
                        socketio.emit('classification', result, room=sid)
                    except Exception as e:
                        print(f"[WS] Classification error: {e}")
                executor.submit(_bg_classify)

            except Exception as e:
                print(f"[WS] Transcription error: {e}")
            finally:
                with _ws_lock:
                    _client_transcribing[sid] = False

        executor.submit(_transcribe_and_emit)

    except Exception as e:
        print(f"[WS] audio_chunk error: {e}")

@app.route('/status', methods=['GET'])
def health_check():
    """
    Health check endpoint to verify the server is running.
    Returns a simple status message.
    """
    return jsonify({"status": "ok"}), 200

@app.route('/vision', methods=['POST'])
def vision_mode():
    """
    Vision Mode endpoint.
    Processes base64-encoded image and returns detected objects with bounding boxes.
    """
    try:
        # Check if JSON data was provided
        if not request.json or 'image' not in request.json:
            return jsonify({"error": "No image data provided"}), 400
        
        # Get base64-encoded image
        image_base64 = request.json['image']
        
        # Decode base64 to bytes
        image_data = base64.b64decode(image_base64)
        
        # Perform object detection
        result = detect_objects(image_data)
        
        return jsonify(result), 200
    
    except Exception as e:
        # Log the error and return a 500 response
        print(f"Error in vision endpoint: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route('/hearing', methods=['POST'])
def hearing_mode():
    """
    Hearing endpoint. Accepts base64-encoded WAV audio and returns transcription and sound classification.
    Runs transcription and classification in parallel for responsiveness.
    """
    temp_path = None
    try:
        # Expect base64-encoded audio in JSON: { "audio": "...base64..." }
        if not request.json or 'audio' not in request.json:
            return jsonify({"error": "Missing audio data in request."}), 400

        audio_base64 = request.json['audio']
        try:
            audio_data = base64.b64decode(audio_base64)
        except Exception:
            return jsonify({"error": "Could not decode audio."}), 400

        if not audio_data or len(audio_data) == 0:
            return jsonify({"error": "Audio data is empty."}), 400

        print(f"[DEBUG] Received audio data: {len(audio_data)} bytes")
        
        # Transcribe directly from memory (no temp file disk I/O)
        try:
            pcm_data = _extract_pcm_from_wav(audio_data)
            if pcm_data:
                pcm_f32 = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
                segments, info = whisper_model.transcribe(
                    pcm_f32,
                    beam_size=1,
                    language="en",
                    condition_on_previous_text=False,
                )
            else:
                # Fallback: save to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                    temp_path = temp_file.name
                    temp_file.write(audio_data)
                segments, info = whisper_model.transcribe(
                    temp_path,
                    beam_size=1,
                    language="en",
                    condition_on_previous_text=False,
                )
        except Exception:
            # Last-resort fallback to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                temp_path = temp_file.name
                temp_file.write(audio_data)
            segments, info = whisper_model.transcribe(
                temp_path,
                beam_size=1,
                language="en",
                condition_on_previous_text=False,
            )
        
        transcription_parts = []
        for segment in segments:
            transcription_parts.append(segment.text.strip())
        full_transcription = " ".join(transcription_parts).strip()
        
        print(f"[DEBUG] Transcription result: '{full_transcription}'")
        print(f"[DEBUG] Language: {info.language} (probability: {info.language_probability:.2f})")
        
        # Clean up temp file immediately
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
            temp_path = None
        
        # --- Sound classification runs in background (non-blocking) ---
        def _bg_classify(data):
            global _latest_classification
            try:
                result = classify_audio(data)
                with _classification_lock:
                    _latest_classification = result
                print(f"[DEBUG] Sound classification: {result}")
                if result.get("alert"):
                    print(f"[ALERT] {result.get('alert')} (confidence: {result.get('confidence'):.2f})")
            except Exception as e:
                print(f"[DEBUG] Classification error: {e}")
        
        executor.submit(_bg_classify, audio_data)
        
        # Return transcription immediately with latest classification result
        with _classification_lock:
            latest = _latest_classification
        
        response = {
            "transcription": full_transcription,
            "alert": latest.get("alert"),
            "alert_type": latest.get("alert_type"),
            "alert_confidence": latest.get("confidence"),
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        # Clean up temp file on error
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass
        
        print(f"Error in hearing endpoint: {str(e)}")
        return jsonify({"error": f"Transcription error: {str(e)}"}), 500

@app.errorhandler(404)
def not_found(error):
    """
    Handle 404 errors with a JSON response.
    """
    return jsonify({"error": "Endpoint not found"}), 404

if __name__ == '__main__':
    print("Atlas Backend Server Starting...")
    print("Server running at: http://127.0.0.1:5000")
    print("Health check: http://127.0.0.1:5000/status")
    print("Vision endpoint: http://127.0.0.1:5000/vision")
    print("Hearing endpoint: http://127.0.0.1:5000/hearing")
    print("WebSocket: ws://127.0.0.1:5000  (live hearing stream)")
    # Use socketio.run for WebSocket + HTTP support
    socketio.run(app, host='127.0.0.1', port=5000, debug=False, allow_unsafe_werkzeug=True)
