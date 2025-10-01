"""
Real-Time Object Detection with YOLOv8
A Streamlit web app that detects objects in uploaded images using YOLOv8.
Features:
- Choose between YOLOv8 Nano, Small, and Medium models
- Adjust confidence and IoU thresholds
- Filter detections by class
- View original and annotated images side by side
- Summary metrics and class counts
- Download annotated image with detections
"""

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import io

# ============================================================
# Page Config
# ============================================================
st.set_page_config(
    page_title="YOLO Object Detector",
    page_icon="🔍",
    layout="wide"
)

# ============================================================
# Load Model (cached so it only loads once)
# ============================================================
@st.cache_resource
def load_model(model_name):
    return YOLO(f"{model_name}.pt")

# ============================================================
# Sidebar Controls
# ============================================================
st.sidebar.title("⚙️ Settings")

model_size = st.sidebar.selectbox(
    "Model Size",
    ["yolov8n", "yolov8s", "yolov8m"],
    index=0,
    help="Nano (fastest) → Small → Medium (most accurate)"
)

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="Only show detections above this confidence"
)

iou_threshold = st.sidebar.slider(
    "IoU Threshold (NMS)",
    min_value=0.1,
    max_value=1.0,
    value=0.45,
    step=0.05,
    help="Non-Maximum Suppression threshold"
)

# Class filter
model = load_model(model_size)
all_classes = list(model.names.values())

selected_classes = st.sidebar.multiselect(
    "Filter Classes (leave empty for all)",
    options=all_classes,
    default=[],
    help="Only show these object types"
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Model Info**")
st.sidebar.markdown(f"Architecture: **{model_size.upper()}**")
st.sidebar.markdown(f"Classes: **{len(all_classes)}** (COCO dataset)")

# ============================================================
# Main Content
# ============================================================
st.title("🔍 Real-Time Object Detection")
st.markdown("Upload an image and **YOLOv8** will detect all objects instantly.")

# File uploader
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp"],
    help="Supports JPG, JPEG, PNG, WEBP"
)

# Sample images option
use_sample = st.checkbox("Or try a sample image")

if use_sample:
    # Use a built-in sample
    sample_url = "https://ultralytics.com/images/bus.jpg"
    st.info(f"Using sample image from YOLO documentation")
    results = model(sample_url, conf=confidence, iou=iou_threshold)
    
    col1, col2 = st.columns(2)
    
    # Original
    orig_img = results[0].orig_img
    orig_img_rgb = orig_img[:, :, ::-1]  # BGR to RGB
    col1.image(orig_img_rgb, caption="Original Image", use_container_width=True)
    
    # Annotated
    annotated = results[0].plot()
    annotated_rgb = annotated[:, :, ::-1]
    col2.image(annotated_rgb, caption="Detections", use_container_width=True)
    
    # Detection details
    boxes = results[0].boxes
    if len(boxes) > 0:
        st.markdown("### Detection Results")
        
        # Summary metrics
        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric("Objects Detected", len(boxes))
        unique_classes = set([model.names[int(c)] for c in boxes.cls])
        mcol2.metric("Unique Classes", len(unique_classes))
        mcol3.metric("Avg Confidence", f"{boxes.conf.mean():.1%}")
        
        # Details table
        for box in boxes:
            cls_name = model.names[int(box.cls)]
            conf_val = float(box.conf)
            st.write(f"**{cls_name}** — {conf_val:.1%}")

elif uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    # Run detection
    with st.spinner("Detecting objects..."):
        results = model(img_array, conf=confidence, iou=iou_threshold)
    
    # Filter by selected classes if any
    if selected_classes:
        class_ids = [k for k, v in model.names.items() if v in selected_classes]
        # Filter boxes
        boxes = results[0].boxes
        mask = torch.tensor([int(c) in class_ids for c in boxes.cls])
        # We'll filter in the display instead
    
    # Display results
    col1, col2 = st.columns(2)
    col1.image(image, caption="Original Image", use_container_width=True)
    
    annotated = results[0].plot()
    annotated_rgb = annotated[:, :, ::-1]  # BGR to RGB
    col2.image(annotated_rgb, caption="Detections", use_container_width=True)
    
    # Detection details
    boxes = results[0].boxes
    
    if len(boxes) > 0:
        st.markdown("### Detection Results")
        
        # Summary metrics
        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric("Objects Detected", len(boxes))
        unique_classes = set([model.names[int(c)] for c in boxes.cls])
        mcol2.metric("Unique Classes", len(unique_classes))
        mcol3.metric("Avg Confidence", f"{boxes.conf.mean():.1%}")
        
        # Count per class
        st.markdown("#### Objects by Class")
        class_counts = {}
        for box in boxes:
            cls_name = model.names[int(box.cls)]
            conf_val = float(box.conf)
            if selected_classes and cls_name not in selected_classes:
                continue
            if cls_name not in class_counts:
                class_counts[cls_name] = {'count': 0, 'max_conf': 0}
            class_counts[cls_name]['count'] += 1
            class_counts[cls_name]['max_conf'] = max(class_counts[cls_name]['max_conf'], conf_val)
        
        for cls_name, info in sorted(class_counts.items(), key=lambda x: -x[1]['count']):
            st.write(f"**{cls_name}**: {info['count']} detected (max conf: {info['max_conf']:.1%})")
        
        # Download annotated image
        st.markdown("---")
        annotated_pil = Image.fromarray(annotated_rgb)
        buf = io.BytesIO()
        annotated_pil.save(buf, format="PNG")
        st.download_button(
            label="📥 Download Annotated Image",
            data=buf.getvalue(),
            file_name="detected.png",
            mime="image/png"
        )
    else:
        st.warning("No objects detected. Try lowering the confidence threshold.")

else:
    # Placeholder
    st.markdown("""
    ### How to Use
    1. **Upload** an image using the file uploader above
    2. **Adjust** settings in the sidebar (model size, confidence)
    3. **View** detections with bounding boxes and confidence scores
    4. **Download** the annotated image
    
    ### Supported Objects
    YOLOv8 detects **80 object classes** from the COCO dataset including: 
    person, car, bus, truck, bicycle, motorcycle, dog, cat, bird, 
    chair, table, laptop, phone, bottle, and many more.
    """)

# Footer
st.markdown("---")
st.markdown(
    "Built with [YOLOv8](https://github.com/ultralytics/ultralytics) + "
    "[Streamlit](https://streamlit.io) | "
    "[GitHub](https://github.com/GamithaManawadu)"
)
