# Road-Damage-Detection-Dataset
A novel dataset with pothole, cracks and manhole, created for robust road-surface damage detection in urban and rural settings.


## Compiling Models for Hailo-8L AI Accelerator

This guide describes the steps to compile YOLOv11 models for deployment on the Hailo-8L AI accelerator hardware.

### Prerequisites

Before starting the compilation process, ensure you have:

1. **Hailo AI Software Suite**: Download from the [Hailo Developer Zone](https://hailo.ai/developer-zone/software-downloads/?product=ai_accelerators&device=hailo_8_8l)
2. **Docker**: Install Docker following the [official Hailo documentation](https://hailo.ai/developer-zone/documentation/hailo-sw-suite-2025-10-for-hailo-8-8l/?sp_referrer=suite/suite_install.html#docker-installation)
3. **Trained ONNX models**: Export your trained YOLOv11 models to ONNX format
4. **Calibration images**: Representative dataset images (we used all) for quantization calibration

### Compilation Steps

#### 1. Launch Docker Container

Start the Hailo AI Software Suite Docker container with HailoRT service enabled:

```bash
./hailo_ai_sw_suite_docker_run.sh --hailort-enable-service
```

#### 2. Compile YOLOv11n Model

Compile the YOLOv11-nano model optimized for performance:

```bash
hailomz compile yolov11n \
  --ckpt=/local/shared_with_docker/best-n.onnx \
  --hw-arch hailo8l \
  --calib-path /local/shared_with_docker/images \
  --classes 3 \
  --performance
```

#### 3. Compile YOLOv11s Model

Compile the YOLOv11-small model optimized for performance:

```bash
hailomz compile yolov11s \
  --ckpt=/local/shared_with_docker/best-s.onnx \
  --hw-arch hailo8l \
  --calib-path /local/shared_with_docker/images \
  --classes 3 \
  --performance
```

### Command Parameters

- `--ckpt`: Path to the ONNX checkpoint file
- `--hw-arch`: Target hardware architecture (hailo8l in this case)
- `--calib-path`: Directory containing calibration images for quantization
- `--classes`: Number of object classes (3 for pothole, crack, maintenance hole)
- `--performance`: Optimization mode prioritizing inference speed

### Output

The compilation process generates HEF (Hailo Executable Format) files optimized for the Hailo-8L accelerator, ready for deployment on edge devices.
