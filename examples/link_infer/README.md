# Example: TrackieLINK C++ Inference

This directory contains a minimal C++ example for running inference with a TrackieLINK model (e.g., `yolov5nu`).

## Building and Running

This example uses CMake to build. It links against ONNX Runtime.

```bash
# Configure the build
cmake -B build -DORT_PATH=/path/to/onnxruntime

# Build the example
cmake --build build

# Run inference
./build/infer_link --model /path/to/yolov5nu.onnx --image /path/to/image.jpg
```

See `main.cpp` for the implementation.
