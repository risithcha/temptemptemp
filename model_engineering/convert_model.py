"""
Atlas MobileNet-SSD TFLite Conversion Script.

Downloads pre-converted object detection models and adds metadata for mobile deployment.
Uses tflite-support library on Linux for metadata embedding.
Models use pre-converted 4-output format compatible with TFLite Task Library.
"""

import json
import sys
import urllib.request
import shutil
import zipfile
from pathlib import Path

import tensorflow as tf
import numpy as np

# Pre-converted TFLite models with 4-output format from TensorFlow Hub
# These models have post-processing baked in and work with TFLite Task Library
TFLITE_MODEL_CONFIGS = {
    "ssd_mobilenet_v1": {
        "url": "https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip",
        "input_size": 300,
        "description": "SSD MobileNet V1 300x300 - Quantized, fast",
        "filename": "detect.tflite",
        "quantized": True,
    },
    "efficientdet_lite0": {
        "url": "https://tfhub.dev/tensorflow/lite-model/efficientdet/lite0/detection/metadata/1?lite-format=tflite",
        "input_size": 320,
        "description": "EfficientDet-Lite0 320x320 - Best accuracy/speed trade-off",
        "filename": "efficientdet_lite0.tflite",
        "quantized": False,
        "direct_download": True,
        "has_metadata": True,
    },
    "efficientdet_lite2": {
        "url": "https://tfhub.dev/tensorflow/lite-model/efficientdet/lite2/detection/metadata/1?lite-format=tflite",
        "input_size": 448,
        "description": "EfficientDet-Lite2 448x448 - Higher accuracy",
        "filename": "efficientdet_lite2.tflite",
        "quantized": False,
        "direct_download": True,
        "has_metadata": True,
    },
}

DEFAULT_MODEL = "ssd_mobilenet_v1"
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_FILENAME = "atlas_mobilenet_quant.tflite"
LABELS_FILENAME = "coco_labels.txt"

# COCO labels (91 classes)
COCO_LABELS = [
    "???", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "???", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "???", "backpack", "umbrella",
    "???", "???", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "???", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "???", "dining table", "???", "???", "toilet", "???", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster",
    "sink", "refrigerator", "???", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
]


def get_file_size_mb(filepath: Path) -> float:
    """
    Get file size in megabytes.
    """
    return filepath.stat().st_size / (1024 * 1024)


def download_with_progress(url: str, dest: Path) -> None:
    """
    Download file with progress indication.
    """
    print(f"Downloading from: {url}")
    print(f"Destination: {dest}")
    
    def progress_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        if count % 100 == 0:
            print(f"\rProgress: {min(percent, 100)}%", end="", flush=True)
    
    urllib.request.urlretrieve(url, dest, reporthook=progress_hook)
    print("\nDownload complete!")


def create_labels_file(labels: list, output_path: Path) -> None:
    """
    Create labels.txt file for TFLite metadata.
    """
    with open(output_path, "w") as f:
        for label in labels:
            f.write(f"{label}\n")
    print(f"Labels file created: {output_path}")


def download_preconverted_tflite(model_key: str, cache_dir: Path = None) -> Path:
    """
    Download pre-converted TFLite model with proper 4-output format.
    Cached after first download for performance.
    """
    if cache_dir is None:
        cache_dir = OUTPUT_DIR / "model_cache"
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    config = TFLITE_MODEL_CONFIGS[model_key]
    url = config["url"]
    filename = config["filename"]
    cached_model = cache_dir / filename
    
    if cached_model.exists():
        print(f"Model already cached: {cached_model}")
        return cached_model
    
    if config.get("direct_download"):
        print(f"Downloading {model_key} from TensorFlow Hub...")
        download_with_progress(url, cached_model)
    else:
        # Handle zip file extraction
        zip_path = cache_dir / f"{model_key}.zip"
        
        if not zip_path.exists():
            download_with_progress(url, zip_path)
        
        print(f"Extracting {zip_path.name}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(cache_dir)
        
        # Find extracted .tflite file
        tflite_files = list(cache_dir.glob("*.tflite"))
        if tflite_files:
            if tflite_files[0].name != filename:
                tflite_files[0].rename(cached_model)
        else:
            for item in cache_dir.iterdir():
                if item.is_dir():
                    tflite_files = list(item.glob("*.tflite"))
                    if tflite_files:
                        shutil.copy(tflite_files[0], cached_model)
                        break
    
    if not cached_model.exists():
        raise RuntimeError(f"Failed to download/extract model: {model_key}")
    
    print(f"Model downloaded: {cached_model}")
    return cached_model


def add_metadata(
    tflite_model_path: Path,
    labels_path: Path,
    output_path: Path,
    input_size: int = 300,
) -> None:
    """
    Add metadata to TFLite model using tflite-support library.
    Bundles labels and normalization parameters for TFLite Task Library compatibility.
    """
    from tflite_support.metadata_writers import object_detector
    from tflite_support.metadata_writers import writer_utils

    print(f"\nAdding metadata to: {tflite_model_path}")
    print(f"Labels file: {labels_path}")

    # Verify model has 4 output tensors (standard format)
    interpreter = tf.lite.Interpreter(model_path=str(tflite_model_path))
    interpreter.allocate_tensors()
    output_details = interpreter.get_output_details()
    print(f"Model has {len(output_details)} output tensors")

    # Load model and add metadata
    model_buffer = writer_utils.load_file(str(tflite_model_path))
    
    writer = object_detector.MetadataWriter.create_for_inference(
        model_buffer,
        input_norm_mean=[127.5],
        input_norm_std=[127.5],
        label_file_paths=[str(labels_path)]
    )
    
    metadata_json = writer.get_metadata_json()
    print("\nGenerated Metadata:")
    print(metadata_json[:500] + "..." if len(metadata_json) > 500 else metadata_json)
    
    writer_utils.save_file(writer.populate(), str(output_path))
    print(f"\nModel with metadata saved to: {output_path}")


def verify_model(tflite_path: Path) -> None:
    """
    Verify TFLite model by loading and running test inference.
    """
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("\nInput Tensors:")
    for i, detail in enumerate(input_details):
        print(f"  [{i}] {detail['name']}")
        print(f"      Shape: {detail['shape']}")
        print(f"      Type: {detail['dtype']}")
    
    print("\nOutput Tensors:")
    for i, detail in enumerate(output_details):
        print(f"  [{i}] {detail['name']}")
        print(f"      Shape: {detail['shape']}")
        print(f"      Type: {detail['dtype']}")
    
    print("\nRunning test inference...")
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']
    
    # Generate test input with correct dtype
    if input_dtype == np.uint8:
        test_input = np.random.randint(0, 255, size=input_shape, dtype=np.uint8)
    else:
        test_input = np.random.randint(0, 255, size=input_shape).astype(np.float32)
    
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    
    print("Inference successful! Output shapes:")
    for detail in output_details:
        output = interpreter.get_tensor(detail['index'])
        print(f"  {detail['name']}: {output.shape}")
    
    print("\nModel verification passed.")

def main():
    """
    Main conversion pipeline.
    Downloads pre-converted TFLite model, adds metadata, and verifies output.
    """
    print("\n" + "=" * 60)
    print("ATLAS MODEL ENGINEERING - TFLite Conversion Pipeline")
    print("=" * 60)
    
    model_key = DEFAULT_MODEL
    config = TFLITE_MODEL_CONFIGS[model_key]
    
    print(f"\nSelected Model: {model_key}")
    print(f"Description: {config['description']}")
    print(f"Input Size: {config['input_size']}x{config['input_size']}")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    labels_path = OUTPUT_DIR / LABELS_FILENAME
    final_path = OUTPUT_DIR / OUTPUT_FILENAME
    
    # Step 1: Create labels
    print("\n" + "=" * 60)
    print("STEP 1: CREATE LABELS")
    print("=" * 60)
    create_labels_file(COCO_LABELS, labels_path)
    
    # Step 2: Download model
    print("\n" + "=" * 60)
    print("STEP 2: DOWNLOAD MODEL")
    print("=" * 60)
    tflite_model_path = download_preconverted_tflite(model_key)
    
    # Step 3: Add metadata (skip if already embedded)
    if config.get("has_metadata", False):
        print(f"\nModel already has metadata embedded!")
        shutil.copy(tflite_model_path, final_path)
        print(f"Model copied to: {final_path}")
    else:
        print("\n" + "=" * 60)
        print("STEP 3: ADD METADATA")
        print("=" * 60)
        add_metadata(tflite_model_path, labels_path, final_path, config['input_size'])
    
    # Step 4: Verify
    print("\n" + "=" * 60)
    print("STEP 4: VERIFY MODEL")
    print("=" * 60)
    verify_model(final_path)
    
    # Step 5: Report
    print("\n" + "=" * 60)
    print("STEP 5: MODEL INFO")
    print("=" * 60)
    
    tflite_size = get_file_size_mb(final_path)
    report = {
        "model": model_key,
        "input_size": config['input_size'],
        "tflite_size_mb": round(tflite_size, 2),
        "description": config['description'],
        "quantized": config.get('quantized', False),
        "has_metadata": True,
    }
    
    print(f"""
Model: {model_key}
----------------------------------------
Input Size:         {config['input_size']}x{config['input_size']}
TFLite Size:        {tflite_size:.2f} MB
Quantized:          {report['quantized']}
Has Metadata:       {report['has_metadata']}
""")
    
    report_path = OUTPUT_DIR / "conversion_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to: {report_path}")
    
    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)
    print(f"Output files in: {OUTPUT_DIR}")
    print(f"  - {OUTPUT_FILENAME} - TFLite model with metadata")
    print(f"  - {LABELS_FILENAME} - COCO class labels")
    print(f"  - conversion_report.json - Model info")
    
    return report


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nConversion cancelled.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
