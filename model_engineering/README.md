# Atlas Model Engineering - TFLite Conversion

This folder contains the AI engineering pipeline for downloading and preparing object detection models in TensorFlow Lite format optimized for mobile deployment.

## Key Change: Using Pre-Converted Models

The script now uses **pre-converted TFLite models** from TensorFlow Hub/Google's model zoo instead of converting TF2 Detection Zoo SavedModels directly.

**Why?** TF2 Detection Zoo SavedModels have **8 raw output tensors** without post-processing, which are incompatible with standard TFLite metadata writers. Pre-converted models have the standard **4-output format** that works with TFLite Task Library.

### Available Models

| Model | Input Size | Speed | Accuracy | Size |
|-------|------------|-------|----------|------|
| `ssd_mobilenet_v1` (default) | 300x300 | Fast | Good | ~4 MB |
| `efficientdet_lite0` | 320x320 | Medium | Better | ~5 MB |
| `efficientdet_lite2` | 448x448 | Slower | Best | ~12 MB |
```

## Overview

The pipeline now:
1. Downloads a **pre-converted TFLite model** (with post-processing baked in)
2. Adds **metadata and labels** if the model does not already include it
3. Verifies inference on random input and writes a short report

Default model is `ssd_mobilenet_v1`. Models with `has_metadata: true` (efficientdet_lite0/2) are simply copied after download.

### Output

The script writes to `output/`:
- `atlas_mobilenet_quant.tflite` — Model with embedded metadata and labels
- `coco_labels.txt` — COCO label map (91-line format)
- `conversion_report.json` — Basic model info (size, input dimensions)

## Technical Details

### Metadata

TFLite metadata tells mobile apps how to use the model:

```json
{
  "input": {
    "shape": [1, 300, 300, 3],
    "normalization": {"mean": 127.5, "std": 127.5}
  },
  "outputs": ["boxes", "classes", "scores", "num_detections"],
  "labels": "coco_labels.txt"
}
```

This enables automatic preprocessing in TensorFlow Lite Task Library.

## Output Tensors

The converted model outputs:

| Tensor | Shape | Description |
|--------|-------|-------------|
| `detection_boxes` | [1, N, 4] | Bounding boxes [y1, x1, y2, x2] normalized to [0,1] |
| `detection_classes` | [1, N] | Class indices (1-90, see COCO labels) |
| `detection_scores` | [1, N] | Confidence scores (0.0-1.0) |
| `num_detections` | [1] | Number of valid detections |

## Data Sources & Verification

### COCO Labels (`coco_labels.txt`)

**Source:** [tensorflow/models/mscoco_label_map.pbtxt](https://github.com/tensorflow/models/blob/master/research/object_detection/data/mscoco_label_map.pbtxt)

The COCO dataset has:
- **91 original category IDs** (from the research paper)
- **80 actual classes** (11 IDs were never released in 2014/2017 datasets)

**Label File Format:** Our `coco_labels.txt` uses the **91-line format** where:
- Line 0 = `???` (background/unused class 0)
- Line N = class name for class ID N
- Lines for missing IDs = `???` placeholder

This format is required because **SSD MobileNet models from TF Object Detection Zoo output original COCO class IDs** (1-90 with gaps), not remapped sequential IDs.

### Normalization Parameters

**Source:** [mobile_ssd_tflite_client.cc](https://github.com/tensorflow/models/blob/master/research/lstm_object_detection/tflite/mobile_ssd_tflite_client.cc#L148-L156)

```cpp
void MobileSSDTfLiteClient::SetImageNormalizationParams() {
  mean_value_ = 127.5f;
  std_value_ = 127.5f;
}
```

**Reasoning:**
- MobileNet expects input pixels normalized to `[-1, 1]` range
- Formula: `normalized = (pixel - 127.5) / 127.5`
- Input 0 → -1.0, Input 127.5 → 0.0, Input 255 → 1.0

**Additional Verification:** [TensorFlow Object Detection TFLite Tutorial](https://github.com/tensorflow/models/blob/master/research/object_detection/colab_tutorials/convert_odt_model_to_TFLite.ipynb)
> "As the SSD MobileNet V2 FPNLite 640x640 model takes input image with pixel value in the range of [-1..1], we need to set `norm_mean = 127.5` and `norm_std = 127.5`."

### Why We Embed Metadata?

| Benefit | Description |
|---------|-------------|
| **Self-contained** | Labels, input specs, normalization bundled in one `.tflite` file |
| **SDK Compatible** | TFLite Task Library can auto-preprocess using embedded metadata |
| **Documentation** | Model file describes its own requirements |
| **Portability** | No need to ship separate config files |

## References

- [TensorFlow Model Garden - Object Detection](https://github.com/tensorflow/models/tree/master/research/object_detection)
- [TFLite Converter Guide](https://ai.google.dev/edge/litert/conversion/tensorflow/convert_tf)
- [Post-Training Quantization](https://ai.google.dev/edge/litert/conversion/tensorflow/quantization/post_training_quantization)
- [TFLite Metadata Writer](https://ai.google.dev/edge/litert/conversion/tensorflow/metadata_writer_tutorial)
- [COCO Dataset Official](https://cocodataset.org/)
- [COCO Label Map (TensorFlow)](https://github.com/tensorflow/models/blob/master/research/object_detection/data/mscoco_label_map.pbtxt)
- [COCO Label Research](https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/)

## License

The MobileNet-SSD models are licensed under Apache 2.0.
COCO dataset annotations are under Creative Commons.
