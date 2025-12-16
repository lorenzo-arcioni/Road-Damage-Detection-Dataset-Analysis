import cv2
import numpy as np
import time
from hailo_platform import (
    HEF, ConfigureParams, FormatType, HailoSchedulingAlgorithm,
    HailoStreamInterface, InputVStreamParams, OutputVStreamParams,
    InferVStreams, VDevice
)

# Parameters
HEF_PATH = "yolov11n.hef"
CONF_THRESHOLD = 0.4
WEBCAM_ID = 0  # Change if you have multiple webcams (0, 1, 2...)

def preprocess_frame(frame, target_size):
    """
    Preprocess the webcam frame for the model.
    - Resize while preserving the aspect ratio.
    - Add padding to match the model input size.
    """
    h, w = frame.shape[:2]
    th, tw = target_size
    
    # Resize while keeping aspect ratio with padding
    scale = min(tw/w, th/h)
    new_w, new_h = int(w*scale), int(h*scale)
    
    resized = cv2.resize(frame, (new_w, new_h))
    
    # Center with padding
    padded = np.zeros((th, tw, 3), dtype=np.uint8)
    y_offset = (th - new_h) // 2
    x_offset = (tw - new_w) // 2
    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    input_tensor = padded.astype(np.float32)[np.newaxis, ...]
    
    return input_tensor, (scale, x_offset, y_offset, w, h)

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
        
        # Convert from normalized coordinates to model pixel coordinates (if needed)
        if x1_model <= 1.0 and y1_model <= 1.0 and x2_model <= 1.0 and y2_model <= 1.0:
            x1_model *= tw
            y1_model *= th
            x2_model *= tw
            y2_model *= th
        
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
    
    return boxes, confidences

def draw_boxes(frame, boxes, confidences):
    """Draw bounding boxes on the frame and add confidence scores."""
    for i, (x, y, w, h) in enumerate(boxes):
        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Draw confidence above the box
        conf_text = f'{confidences[i]:.2f}'
        (w_text, h_text), _ = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x, y - h_text - 10), (x + w_text, y), (0, 255, 0), -1)
        cv2.putText(frame, conf_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    return frame

def main():
    """Main function for running the real-time inference pipeline."""
    # Hailo setup
    print("Initializing Hailo...")
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
    print(f"Confidence threshold: {CONF_THRESHOLD}")
    
    # Webcam setup
    print(f"Opening webcam {WEBCAM_ID}...")
    cap = cv2.VideoCapture(WEBCAM_ID)
    
    if not cap.isOpened():
        print("ERROR: Unable to open the webcam!")
        target.release()
        return
    
    print("Webcam opened successfully!")
    print("Press 'q' to quit")
    
    # Setup inference pipeline
    input_params = InputVStreamParams.make(network_group, quantized=False, format_type=FormatType.FLOAT32)
    output_params = OutputVStreamParams.make(network_group, quantized=False, format_type=FormatType.FLOAT32)
    
    with InferVStreams(network_group, input_params, output_params) as pipeline:
        with network_group.activate(network_group.create_params()):
            
            frame_count = 0
            
            while True:
                # Read frame from webcam
                ret, frame = cap.read()
                if not ret:
                    print("Error reading frame")
                    break
                
                frame_count += 1
                
                # Preprocess the frame
                input_tensor, preproc_info = preprocess_frame(frame, target_size)
                
                # Inference
                inference_start = time.time()
                result = pipeline.infer({input_info.name: input_tensor})
                inference_time = (time.time() - inference_start) * 1000
                
                # Postprocess
                output = result[output_info.name][0][0]
                boxes, confidences = parse_output_and_transform(output, target_size, preproc_info, CONF_THRESHOLD)
                
                # Print everything on a single line
                detection_info = " | ".join([f"Conf:{conf:.4f} Box:({x},{y},{w},{h})" 
                                             for (x,y,w,h), conf in zip(boxes, confidences)])
                if len(boxes) == 0:
                    detection_info = "No detections"
                
                print(f"Frame {frame_count} | Inference: {inference_time:.2f}ms | Detections: {len(boxes)} | {detection_info}")
                
                # Draw bounding boxes on the original frame
                frame_with_boxes = draw_boxes(frame, boxes, confidences)
                
                # Add FPS and detection info
                fps_text = f'Detections: {len(boxes)} | Inference: {inference_time:.1f}ms'
                cv2.putText(frame_with_boxes, fps_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Show the frame
                cv2.imshow('Hailo Real-time Detection', frame_with_boxes)
                
                # Exit when pressing 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
    # Cleanup
    print("\nShutting down...")
    cap.release()
    cv2.destroyAllWindows()
    target.release()
    print("Done!")

if __name__ == "__main__":
    main()

