# -*- coding: utf-8 -*-
"""
Drone Footage Object Detection and Tracking App
Fixed version with class names on bounding boxes
"""

# -*- coding: utf-8 -*-
"""
Drone Footage Object Detection and Tracking App
Fixed version with class names on bounding boxes
"""

import streamlit as st
import numpy as np
import tempfile
import os
import cv2
from PIL import Image
from collections import Counter
import time
import torch
import uuid
import shutil
import gc
import traceback
import psutil
import threading
from contextlib import contextmanager

# Import with error handling
try:
    from ultralytics import YOLOWorld
    from sahi.predict import get_sliced_prediction
    from sahi.models.ultralytics import UltralyticsDetectionModel
    from deep_sort_realtime.deepsort_tracker import DeepSort
except ImportError as e:
    st.error(f"Missing required package: {e}")
    st.stop()

# Memory management context manager
@contextmanager
def memory_cleanup():
    try:
        yield
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Streamlit Page Config
st.set_page_config(page_title='Drone Footage Object Detector', page_icon='🖈', layout='wide')

# Add memory info to sidebar
def show_memory_info():
    """Display current memory usage"""
    memory = psutil.virtual_memory()
    st.sidebar.markdown("---")
    st.sidebar.markdown("**System Info**")
    st.sidebar.markdown(f"Memory: {memory.percent}% used")
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024**3
        st.sidebar.markdown(f"GPU Memory: {gpu_memory:.1f} GB")

# Styling
st.markdown("""
    <style>
        .main {max-height: 100vh; overflow-y: scroll;}
        .block-container {
            padding-top: 1rem !important; 
            margin-top: 0rem !important;
            max-width: 100% !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
        .title {
            font-size: 32px; 
            font-family: Georgia, serif; 
            text-align: center;
            color: #FFFFFF;
            margin-bottom: 1rem;
            padding: 10px;
            word-wrap: break-word;
            line-height: 1.2;
            width: 100%;
            max-width: 100%;
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(to bottom, #0F202B, #202D4A, #172626);
            padding: 20px;
            border-right: 2px solid #FFFFFF;
            font-family: Verdana, sans-serif;
            font-size: 16px;
            width: 300px !important;
        }
        div[data-testid="stSidebarContent"] {width: 100% !important;}
        main {
            background: linear-gradient(to bottom right, #0F202B, #020229); 
            padding: 10px;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: rgba(255, 255, 255, 0.1);
            border-radius: 4px 4px 0px 0px;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #8cc8e6;
            color: #000000;
        }
        
        /* Responsive title for smaller screens */
        @media (max-width: 768px) {
            .title {
                font-size: 24px;
                padding: 5px;
            }
        }
        
        @media (max-width: 480px) {
            .title {
                font-size: 20px;
                line-height: 1.1;
            }
        }
    </style>
""", unsafe_allow_html=True)

# Create a container for the title to ensure full width
title_container = st.container()
with title_container:
    st.markdown(
        '<div class="title">Drone Footage Object Detection and Tracking</div>', 
        unsafe_allow_html=True
    )
tab1, tab2 = st.tabs(["📸 Image                    ", "🎥 Video                  "])

# Sidebar Configuration
st.sidebar.markdown("## Model Configuration")

selected_model = st.sidebar.radio(
    label="Select Detection Mode",
    options=["Default Detection", "Text-prompt Detection"],
    index=0,
    help="Select desired detection option"
)

confidence_value = st.sidebar.slider(
    "Select model confidence value", 
    min_value=0.1, 
    max_value=1.0, 
    value=0.25, 
    step=0.05
)

# Video processing settings
st.sidebar.markdown("---")
st.sidebar.markdown("**Video Processing Settings**")

skip_frames = st.sidebar.slider(
    "Frame skip (higher = faster, lower accuracy)", 
    min_value=1, 
    max_value=10, 
    value=3, 
    step=1,
    help="Process every Nth frame to speed up processing"
)

max_file_size = st.sidebar.slider(
    "Max video file size (MB)", 
    min_value=10, 
    max_value=200, 
    value=50, 
    step=10,
    help="Reduced default for stability"
)

# Model paths
model_path = "last.pt"
text_prompt_model_path = "yolov8l-worldv2.pt"

# Validate model files exist
if not os.path.exists(model_path):
    st.sidebar.warning(f"❗Default model file '{model_path}' not found!")

if selected_model == "Text-prompt Detection" and not os.path.exists(text_prompt_model_path):
    st.sidebar.warning(f"❗Text prompt model file '{text_prompt_model_path}' not found!")

category_names = []
if selected_model == "Text-prompt Detection":
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Text Prompt Settings**")
    user_prompts = st.sidebar.text_input(
        "Enter class names (separated by comma):",
        help="Enter the objects you want to detect, separated by commas",
        placeholder="person, car, drone"
    )
    category_names = [x.strip() for x in user_prompts.split(",") if x.strip()]
    
    if category_names:
        st.sidebar.write("**Classes to detect:**")
        for i, name in enumerate(category_names, 1):
            st.sidebar.write(f"{i}. {name}")

# Show system info
show_memory_info()

# Image Processing Functions (keeping original logic)
def run_sahi_yolo_inference(image_pil, model_path, conf):
    """Run SAHI inference with default YOLO model"""
    try:
        with memory_cleanup():
            image_np = np.array(image_pil.convert("RGB"))
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            detection_model = UltralyticsDetectionModel(
                model_path=model_path,
                confidence_threshold=conf,
                device="cuda:0" if torch.cuda.is_available() else "cpu"
            )
            
            result = get_sliced_prediction(
                image_np,
                detection_model,
                slice_height=512,
                slice_width=512,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2
            )
            return result
    except Exception as e:
        st.error(f"Error in SAHI inference: {str(e)}")
        return None

def run_text_prompt_sahi_inference(image_pil, model_path, conf, category_names):
    """Text prompt SAHI inference with YOLO World"""
    try:
        with memory_cleanup():
            image_np = np.array(image_pil.convert("RGB"))
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            model = YOLOWorld(model_path)
            model.set_classes(category_names)
            
            detection_model = UltralyticsDetectionModel(
                model=model,
                confidence_threshold=conf,
                device="cuda:0" if torch.cuda.is_available() else "cpu"
            )
            
            result = get_sliced_prediction(
                image_np,
                detection_model,
                slice_height=256,
                slice_width=256,
                overlap_height_ratio=0.3,
                overlap_width_ratio=0.3
            )
            return result
    except Exception as e:
        st.error(f"Error in text prompt inference: {str(e)}")
        return None

# Tab 1: Image Processing (keeping original logic)
with tab1:
    st.markdown(
        '<p style="font-size:22px; font-family:\'Segoe UI\', sans-serif; font-weight:bold; color:#8cc8e6; margin-top:2px;">📸 Upload a drone image</p>',
        unsafe_allow_html=True
    )
    
    uploaded_image = st.file_uploader(
        "Upload Image", 
        type=['jpg', 'jpeg', 'png', 'webp'], 
        key="img_upload"
    ) 
    
    if uploaded_image is not None:
        try:
            image = Image.open(uploaded_image)
            st.markdown("### Uploaded Image Preview")
            st.image(image, caption="Uploaded Image", use_container_width=True)

            if selected_model == "Text-prompt Detection" and not category_names:
                st.warning("❗Please enter class names in the sidebar for text prompt detection!")
            else:
                if selected_model == "Text-prompt Detection" and category_names:
                    st.info(f"**Detecting:** {', '.join(category_names)}")

                # Run inference
                with st.spinner("Running SAHI tiled inference..."):
                    result = None
                    result_img_path = None
                    
                    if selected_model == "Default Detection":
                        result = run_sahi_yolo_inference(image, model_path, confidence_value)
                    else:
                        result = run_text_prompt_sahi_inference(
                            image, text_prompt_model_path, confidence_value, category_names
                        )
                    
                    if result is None:
                        st.error("❌ Inference failed. Please check your model files and try again.")
                    else:
                        unique_img_name = f"result_{uuid.uuid4().hex}"
                        output_dir = os.path.abspath("outputs")
                        os.makedirs(output_dir, exist_ok=True)
                        result_img_path = os.path.join(output_dir, f"{unique_img_name}.png")

                        try:
                            result.export_visuals(
                                export_dir=output_dir,   
                                file_name=unique_img_name,
                                text_size=0.5,
                                rect_th=1,
                                hide_labels=False,
                                hide_conf=True,
                            )
                            time.sleep(1)
                            
                            if os.path.exists(result_img_path):
                                st.success("✔ Inference completed and image exported successfully!")
                            else:
                                st.error("❌ Failed to export result visualization.")
                                result_img_path = None
                                
                        except Exception as e:
                            st.error(f"Failed to export result visualization: {e}")
                            result_img_path = None

                # Display results
                if result_img_path and os.path.exists(result_img_path):
                    st.markdown("### 🎯 Detection Results")
                    
                    with open(result_img_path, 'rb') as f:
                        img_bytes = f.read()
                    st.image(img_bytes, caption="Detected Objects with SAHI", use_container_width=True)
                    
                    st.download_button(
                        label="Download Annotated Image",
                        data=img_bytes,
                        file_name=f"detected_{uploaded_image.name}",
                        mime="image/png"
                    )
                    
                    st.markdown("### Detection Summary")
                    if hasattr(result, 'object_prediction_list') and result.object_prediction_list:
                        class_names = [pred.category.name for pred in result.object_prediction_list]
                        class_counts = Counter(class_names)
                        
                        for cls, count in class_counts.items():
                            st.markdown(f"- **{cls}**: {count}")
                        
                        st.markdown(f"**Total objects detected:** {len(result.object_prediction_list)}")
                    else:
                        st.markdown("No objects detected in the image.")
                elif result is not None:
                    st.warning("❗Detection completed but could not save the result image.")
                        
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

# Fixed Video Processing Function with Class Names on Bounding Boxes
def process_video_with_yolo_deepsort_robust(video_path, output_path, conf, selected_model, category_names=None, skip_frames=3):
    """Robust video processing with class names displayed on bounding boxes"""
    
    cap = None
    out = None
    model = None
    tracker = None
    
    try:
        # Initialize model with error handling
        if selected_model == "Default Detection":
            if not os.path.exists(model_path):
                return None, None, "Model file not found"
            model = YOLOWorld(model_path)
        else:
            if not category_names:
                return None, None, "No category names provided"
            if not os.path.exists(text_prompt_model_path):
                return None, None, "Text prompt model file not found"
            model = YOLOWorld(text_prompt_model_path)
            model.set_classes(category_names)
        
        # Move model to CPU for stability (GPU can cause crashes)
        device = "cpu"  # Force CPU for stability
        model.model.to(device)
        
        # Initialize tracker
        tracker = DeepSort(max_age=10, n_init=3)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, None, "Could not open video file"
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps <= 0 or np.isnan(fps):
            fps = 24.0
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            return None, None, "Could not initialize video writer"
        
        frame_count = 0
        processed_frames = 0
        detection_counts = Counter()
        prev_tracks = []
        track_class_map = {}  # Store class info for each track
        
        # Create progress containers
        progress_container = st.container()
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
            frame_info = st.empty()
        
        # Process frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Update progress every 10 frames
            if frame_count % 10 == 0:
                progress = min(frame_count / max(total_frames, 1), 1.0)
                progress_bar.progress(progress)
                status_text.text(f"Processing: {frame_count}/{total_frames} frames")
                frame_info.text(f"Processed: {processed_frames} | Skipped: {frame_count - processed_frames}")
            
            # Run detection every skip_frames
            if frame_count % skip_frames == 0:
                processed_frames += 1
                try:
                    # Run inference with timeout protection
                    with memory_cleanup():
                        results = model.predict(
                            frame, 
                            conf=conf, 
                            iou=0.6, 
                            augment=False, 
                            verbose=False,
                            imgsz=640  # Smaller size for stability
                        )
                    
                    if results and len(results) > 0:
                        results = results[0]
                        
                        detections = []
                        current_detections_info = []  # Store class info for current detections
                        
                        if results.boxes is not None and len(results.boxes) > 0:
                            for box in results.boxes:
                                try:
                                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                                    conf_score = float(box.conf[0].cpu().numpy())
                                    cls = int(box.cls[0].cpu().numpy())
                                    
                                    # Get class name
                                    if selected_model == "Default Detection":
                                        class_name = model.names.get(cls, f"class_{cls}")
                                    else:
                                        class_name = category_names[cls] if cls < len(category_names) else f"class_{cls}"
                                    
                                    detection_counts[class_name] += 1
                                    detections.append(([x1, y1, x2 - x1, y2 - y1], conf_score, cls))
                                    current_detections_info.append({
                                        'bbox': [x1, y1, x2 - x1, y2 - y1],
                                        'class_name': class_name,
                                        'confidence': conf_score
                                    })
                                except Exception:
                                    continue  # Skip invalid detections
                        
                        # Update tracker
                        tracks = tracker.update_tracks(detections, frame=frame)
                        
                        # Update track-class mapping
                        for i, track in enumerate(tracks):
                            if track.is_confirmed() and i < len(current_detections_info):
                                track_class_map[track.track_id] = current_detections_info[i]['class_name']
                        
                        prev_tracks = tracks
                    else:
                        tracks = prev_tracks
                        
                except Exception:
                    # Use previous tracks if detection fails
                    tracks = prev_tracks
            else:
                tracks = prev_tracks
            
            # Draw tracking results with class names
            try:
                for track in tracks:
                    if not track.is_confirmed():
                        continue
                        
                    ltrb = track.to_ltrb()
                    l, t, r, b = map(int, ltrb)
                    track_id = track.track_id
                    
                    # Get class name for this track
                    class_name = track_class_map.get(track_id, "Unknown")
                    
                    # Choose color based on class (you can customize this)
                    colors = {
                        'person': (0, 255, 0),      # Green
                        'car': (255, 0, 0),         # Blue
                        'truck': (0, 0, 255),       # Red
                        'drone': (255, 255, 0),     # Cyan
                        'bike': (255, 0, 255),      # Magenta
                        'motorcycle': (128, 0, 128), # Purple
                    }
                    color = colors.get(class_name.lower(), (0, 255, 255))  # Default yellow
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (l, t), (r, b), color, 2)
                    
                    # Create label with class name and track ID
                    label = f"{class_name} ID:{track_id}"
                    
                    # Calculate text size for background
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.6
                    thickness = 2
                    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                    
                    # Draw background rectangle for text
                    cv2.rectangle(frame, (l, t - text_height - 10), 
                                (l + text_width, t), color, -1)
                    
                    # Draw text
                    cv2.putText(frame, label, (l, t - 5), 
                               font, font_scale, (255, 255, 255), thickness)
                               
            except Exception:
                pass  # Continue even if drawing fails
            
            # Write frame
            out.write(frame)
            
            # Memory cleanup every 100 frames
            if frame_count % 100 == 0:
                gc.collect()
        
        # Clear progress indicators
        progress_container.empty()
        
        return output_path, detection_counts, None
        
    except Exception as e:
        error_msg = f"Video processing error: {str(e)}"
        return None, None, error_msg
        
    finally:
        # Cleanup resources
        try:
            if cap is not None:
                cap.release()
            if out is not None:
                out.release()
        except:
            pass
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Tab 2: Robust Video Processing
with tab2:
    st.markdown(
        '<p style="font-size:22px; font-family:\'Segoe UI\', sans-serif; font-weight:bold; color:#8cc8e6; margin-top:2px;">🎥 Upload a drone video</p>',
        unsafe_allow_html=True
    )
    
    st.markdown("**Select a video file to upload:**")
    uploaded_video = st.file_uploader(
        "Choose a video file", 
        type=['mp4', 'avi', 'mov', 'mkv'], 
        key="vid_upload",
        help="Supported formats: MP4, AVI, MOV, MKV"
    )
    
    st.info(f"📁 Maximum file size: {max_file_size} MB | Frame skip: {skip_frames}")
    
   
    if uploaded_video is not None:
        file_size = len(uploaded_video.getvalue()) / (1024 * 1024)
        st.success(f"✅ Video uploaded! Size: {file_size:.1f} MB")
        
        if file_size > max_file_size:
            st.error(f"❗File too large! Limit: {max_file_size}MB")
            st.info("Increase the limit in sidebar or use a smaller video")
        else:
            # Show video preview
            st.markdown("### Video Preview")
            try:
                st.video(uploaded_video)
            except:
                st.warning("Could not display preview, but processing should still work")
            
            # Show file info
            st.markdown(f"**File:** {uploaded_video.name} | **Size:** {file_size:.1f} MB")
            
            # Model validation
            if selected_model == "Text-prompt Detection" and not category_names:
                st.warning("❗Enter class names in sidebar for text-prompt detection!")
            elif selected_model == "Text-prompt Detection":
                st.info(f"**Detecting:** {', '.join(category_names)}")
            else:
                st.info("**Using default YOLO model**")
            
            st.markdown("---")
            
            # Processing button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("Process Video", key="process_btn", use_container_width=True):
                    
                    # Create temp files
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
                        tmp_in.write(uploaded_video.read())
                        temp_input = tmp_in.name
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix="_out.mp4") as tmp_out:
                        temp_output = tmp_out.name
                    
                    try:
                        st.markdown("### 🔄 Processing Video...")
                        st.info("This may take several minutes. Please don't refresh the page.")
                        
                        # Process video
                        result_path, detection_counts, error_msg = process_video_with_yolo_deepsort_robust(
                            temp_input,
                            temp_output,
                            confidence_value,
                            selected_model,
                            category_names,
                            skip_frames
                        )
                        
                        if error_msg:
                            st.error(f"❌ Processing failed: {error_msg}")
                        elif result_path and os.path.exists(result_path) and os.path.getsize(result_path) > 1000:
                            st.success("✔ Video processed successfully!")
                            
                            # Display result
                            st.markdown("### 🎯 Processed Video")
                            with open(result_path, 'rb') as f:
                                video_bytes = f.read()
                            
                            st.video(video_bytes)
                            
                            # Download button
                            st.download_button(
                                "Download Processed Video",
                                data=video_bytes,
                                file_name=f"processed_{uploaded_video.name}",
                                mime="video/mp4"
                            )
                            
                            # Show detection summary
                            st.markdown("### Detection Summary")
                            if detection_counts:
                                total = sum(detection_counts.values())
                                st.markdown(f"**Total detections:** {total}")
                                for cls, count in detection_counts.most_common():
                                    st.markdown(f"- **{cls}:** {count}")
                            else:
                                st.markdown("No objects detected")
                        else:
                            st.error("❌ Processing failed - output file is empty or missing")
                            
                    except Exception as e:
                        st.error(f"❌ Unexpected error: {str(e)}")
                        st.code(traceback.format_exc())
                        
                    finally:
                        # Cleanup
                        try:
                            os.unlink(temp_input)
                            os.unlink(temp_output)
                        except:
                            pass

# Footer
st.markdown("---")
st.markdown(
    '<p style="text-align: center; color: #8cc8e6; font-size: 14px;">Drone Detection Dashboard | Built with Streamlit & YOLO</p>',
    unsafe_allow_html=True
)
