import cv2
import numpy as np
from hailo_platform import (
    HEF, ConfigureParams, FormatType, HailoSchedulingAlgorithm,
    HailoStreamInterface, InputVStreamParams, OutputVStreamParams,
    InferVStreams, VDevice
)

# Parameters
IMAGE_PATH = "file-16089009.jpg"
OUTPUT_PATH = "out-pothole.jpeg"
HEF_PATH = "yolov11n.hef"
CONF_THRESHOLD = 0.5

def preprocess_image(image_path, target_size):
    """
    Load and preprocess the image for the model.
    - Resize while preserving aspect ratio.
    - Add padding to match the model input size.
    """
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    th, tw = target_size
    
    # Resize while keeping aspect ratio with padding
    scale = min(tw/w, th/h)
    new_w, new_h = int(w*scale), int(h*scale)
    
    resized = cv2.resize(img, (new_w, new_h))
    
    # Center with padding
    padded = np.zeros((th, tw, 3), dtype=np.uint8)
    y_offset = (th - new_h) // 2
    x_offset = (tw - new_w) // 2
    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    input_tensor = padded.astype(np.float32)[np.newaxis, ...]
    
    return img, padded, input_tensor, (scale, x_offset, y_offset, w, h)

def parse_output_and_transform(output, target_size, preproc_info, conf_threshold):
    """
    Parse the model output and convert bounding boxes.
    - The model returns coordinates in the format (x_min, y_min, x_max, y_max, confidence).
    - This function converts them to the original image coordinates.
    """
    h_model, w_model = target_size
    th, tw = h_model, w_model
    scale, x_offset, y_offset, orig_w, orig_h = preproc_info
    
    boxes = []
    confidences = []
    
    for i, detection in enumerate(output):
        # Model output format: (x1, y1, x2, y2, conf)
        if len(detection) < 5 or detection[4] < conf_threshold:
            continue
            
        y1_model, x1_model, y2_model, x2_model, conf = detection[:5]
        
        print(f"Detection {i}: [{x1_model:.3f}, {y1_model:.3f}, {x2_model:.3f}, {y2_model:.3f}] conf={conf:.3f}")
        
        # Convert from normalized coordinates to model pixel coordinates (if needed)
        if x1_model <= 1.0 and y1_model <= 1.0 and x2_model <= 1.0 and y2_model <= 1.0:
            x1_model *= tw
            y1_model *= th
            x2_model *= tw
            y2_model *= th
        
        print(f"  In model (pixels): ({x1_model:.1f},{y1_model:.1f}) to ({x2_model:.1f},{y2_model:.1f})")
        
        # Convert from model coordinates to original image coordinates
        # 1. Remove padding offset
        x1_model -= x_offset
        y1_model -= y_offset
        x2_model -= x_offset
        y2_model -= y_offset
        
        # 2. Scale back to original dimensions
        x1_orig = x1_model / scale
        y1_orig = y1_model / scale
        x2_orig = x2_model / scale
        y2_orig = y2_model / scale
        
        print(f"  In original image: ({x1_orig:.1f},{y1_orig:.1f}) to ({x2_orig:.1f},{y2_orig:.1f})")
        
        # 3. Clip to original image size
        x1_orig = max(0, min(x1_orig, orig_w))
        y1_orig = max(0, min(y1_orig, orig_h))
        x2_orig = max(0, min(x2_orig, orig_w))
        y2_orig = max(0, min(y2_orig, orig_h))
        
        # 4. Convert to (x, y, width, height) format for OpenCV
        x = int(x1_orig)
        y = int(y1_orig)
        w = int(x2_orig - x1_orig)
        h = int(y2_orig - y1_orig)
        
        if w > 0 and h > 0:
            boxes.append([x, y, w, h])
            confidences.append(conf)
            print(f"  Final box: ({x},{y}) size=({w}x{h})")
    
    return boxes, confidences

def draw_boxes(img, boxes, confidences):
    """Draw bounding boxes on the image and add confidence scores."""
    img_copy = img.copy()
    for i, (x, y, w, h) in enumerate(boxes):
        cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img_copy, f'{confidences[i]:.2f}', (x, y-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img_copy

def main():
    """Main function to run the inference pipeline."""
    # Hailo setup
    target = VDevice(VDevice.create_params())
    hef = HEF(HEF_PATH)
    config = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
    network_group = target.configure(hef, config)[0]
    
    # Get model info
    input_info = hef.get_input_vstream_infos()[0]
    output_info = hef.get_output_vstream_infos()[0]
    h_model, w_model = input_info.shape[:2]
    target_size = (h_model, w_model)
    
    print(f"Model dimensions: {target_size}")
    
    # Preprocess - now also returns the original image
    original_img, processed_img, input_tensor, preproc_info = preprocess_image(IMAGE_PATH, target_size)
    
    scale, x_offset, y_offset, orig_w, orig_h = preproc_info
    print(f"Scale: {scale:.3f}, Offset: ({x_offset}, {y_offset}), Orig size: ({orig_w}x{orig_h})")
    
    # Save preprocessed image for debugging
    cv2.imwrite("debug_preprocessed.jpg", processed_img)
    
    # Inference
    input_params = InputVStreamParams.make(network_group, quantized=False, format_type=FormatType.FLOAT32)
    output_params = OutputVStreamParams.make(network_group, quantized=False, format_type=FormatType.FLOAT32)
    
    with InferVStreams(network_group, input_params, output_params) as pipeline:
        with network_group.activate(network_group.create_params()):
            result = pipeline.infer({input_info.name: input_tensor})
    
    # Postprocess - now correctly transforms coordinates
    output = result[output_info.name][0][0]
    print(f"Output shape: {output.shape if hasattr(output, 'shape') else len(output)}")
    
    boxes, confidences = parse_output_and_transform(output, target_size, preproc_info, CONF_THRESHOLD)
    
    # Draw boxes on the ORIGINAL image
    img_with_boxes = draw_boxes(original_img, boxes, confidences)
    cv2.imwrite(OUTPUT_PATH, img_with_boxes)
    print(f"Found {len(boxes)} detections, saved to {OUTPUT_PATH}")
    
    # img_with_boxes = draw_boxes(original_img, boxes, confidences)
    
    target.release()

if __name__ == "__main__":
    main()

