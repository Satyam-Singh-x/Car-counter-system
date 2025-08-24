import streamlit as st
import cv2
import numpy as np
import math
from ultralytics import YOLO
from sort import Sort
import tempfile
import os
from PIL import Image
import time
import cvzone
from typing import List, Tuple, Optional

# Set page configuration
st.set_page_config(
    page_title="Car Counter App",
    page_icon="🚗",
    layout="wide"
)

# Initialize session state
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'processed_video' not in st.session_state:
    st.session_state.processed_video = None
if 'total_count' not in st.session_state:
    st.session_state.total_count = 0
if 'counting_line' not in st.session_state:
    st.session_state.counting_line = [50, 400, 800, 400]  # Default line position

# Class names
classnames = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]


class VehicleCounter:
    def __init__(self):
        self.model = YOLO('yolov8n.pt')
        self.tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
        self.counted_ids = set()
        self.total_count = 0
        self.min_confidence = 0.3
        self.vehicle_classes = ["car", "truck", "bus", "motorcycle"]

    def set_counting_line(self, line_coords: List[int]):
        """Set the counting line coordinates [x1, y1, x2, y2]"""
        self.counting_line = line_coords

    def process_frame(self, frame: np.ndarray, car_icon: Optional[np.ndarray] = None) -> Tuple[np.ndarray, int]:
        """Process a single frame and return annotated frame with count"""
        if car_icon is not None:
            frame = cvzone.overlayPNG(frame, car_icon, (0, 0))

        # Create copy for detection (black out top area for icon)
        frame_for_detection = frame.copy()
        frame_for_detection[0:180, 0:480] = 0

        # Run detection
        try:
            results = self.model(frame_for_detection, stream=True, verbose=False)
            detections = self._extract_detections(results)

            # Update tracker
            if len(detections) > 0:
                tracked_objects = self.tracker.update(detections)
            else:
                tracked_objects = np.empty((0, 5))

            # Process counting
            self._count_vehicles(tracked_objects)

            # Draw annotations
            annotated_frame = self._draw_annotations(frame, tracked_objects)

            return annotated_frame, self.total_count

        except Exception as e:
            st.error(f"Error processing frame: {e}")
            return frame, self.total_count

    def _extract_detections(self, results) -> np.ndarray:
        """Extract detections from YOLO results"""
        detections = np.empty((0, 5))

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                if cls < len(classnames):
                    class_name = classnames[cls]

                    if class_name in self.vehicle_classes and conf > self.min_confidence:
                        detections = np.vstack([detections, [x1, y1, x2, y2, conf]])

        return detections

    def _count_vehicles(self, tracked_objects: np.ndarray):
        """Count vehicles crossing the line"""
        if not hasattr(self, 'counting_line'):
            return

        x1, y1, x2, y2 = self.counting_line

        for obj in tracked_objects:
            if len(obj) < 5:
                continue

            obj_id = int(obj[4])
            cx = int((obj[0] + obj[2]) / 2)  # Center x
            cy = int((obj[1] + obj[3]) / 2)  # Center y

            # Check if center point crosses the line
            if (min(x1, x2) <= cx <= max(x1, x2) and
                    min(y1, y2) - 15 <= cy <= min(y1, y2) + 15):
                if obj_id not in self.counted_ids:
                    self.counted_ids.add(obj_id)
                    self.total_count += 1

    def _draw_annotations(self, frame: np.ndarray, tracked_objects: np.ndarray) -> np.ndarray:
        """Draw annotations on the frame"""
        # Draw counting line
        if hasattr(self, 'counting_line'):
            x1, y1, x2, y2 = self.counting_line
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)

        # Draw bounding boxes and IDs
        for obj in tracked_objects:
            if len(obj) < 5:
                continue

            x1, y1, x2, y2, obj_id = map(int, obj[:5])
            w, h = x2 - x1, y2 - y1

            # Draw bounding box with cvzone style
            cvzone.cornerRect(frame, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))

            # Draw ID with cvzone style
            cv2.putText(frame, f'{obj_id}', (max(0, x1), max(35, y1)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

            # Check if vehicle is crossing the line
            if hasattr(self, 'counting_line'):
                cx = x1 + w // 2
                cy = y1 + h // 2
                lx1, ly1, lx2, ly2 = self.counting_line

                if (min(lx1, lx2) <= cx <= max(lx1, lx2) and
                        min(ly1, ly2) - 15 <= cy <= min(ly1, ly2) + 15):
                    if obj_id in self.counted_ids:
                        cv2.line(frame, (lx1, ly1), (lx2, ly2), (0, 255, 0), 5)

        # Draw total count
        cv2.putText(frame, str(self.total_count), (255, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (50, 50, 255), 5)

        return frame


# Function to load and prepare the car icon
def load_car_icon(icon_path: str) -> np.ndarray:
    """Load and prepare the car icon for overlay"""
    # Check if the icon file exists
    if not os.path.exists(icon_path):
        st.warning(f"Icon file not found at {icon_path}. Using default icon.")
        # Create a simple default icon
        icon = np.zeros((150, 450, 4), dtype=np.uint8)
        cv2.putText(icon, "CAR COUNTER", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255, 255), 3)
        return icon

    # Load the icon with alpha channel
    try:
        imGraphics = cv2.imread(icon_path, cv2.IMREAD_UNCHANGED)

        # Resize the icon
        new_width = 450
        new_height = 150
        imGraphics = cv2.resize(imGraphics, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Ensure it has an alpha channel
        if imGraphics.shape[2] == 3:  # If no alpha channel, add one
            b, g, r = cv2.split(imGraphics)
            alpha = 255 * np.ones(b.shape, dtype=b.dtype)
            imGraphics = cv2.merge([b, g, r, alpha])

        return imGraphics
    except Exception as e:
        st.error(f"Error loading icon: {e}")
        # Create a default icon as fallback
        icon = np.zeros((150, 450, 4), dtype=np.uint8)
        cv2.putText(icon, "CAR COUNTER", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255, 255), 3)
        return icon


# Function to process video
def process_video(input_video_path: str, output_video_path: str, icon_path: str,
                  counting_line: List[int], frame_skip: int = 0) -> int:
    """Process video and return total vehicle count"""
    # Load car icon
    car_icon = load_car_icon(icon_path)

    # Initialize vehicle counter
    counter = VehicleCounter()
    counter.set_counting_line(counting_line)

    # Open input video
    try:
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise Exception("Could not open video file")
    except Exception as e:
        st.error(f"Error opening video: {e}")
        return 0

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Adjust FPS if skipping frames
    output_fps = fps // (frame_skip + 1) if frame_skip > 0 else fps

    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, output_fps, (width, height))

    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    frame_count = 0
    processed_frames = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Skip frames if requested
            if frame_skip > 0 and frame_count % (frame_skip + 1) != 0:
                continue

            processed_frames += 1
            progress = processed_frames / (total_frames // (frame_skip + 1) if frame_skip > 0 else total_frames)
            progress_bar.progress(min(progress, 1.0))
            status_text.text(f"Processing frame {processed_frames}")

            # Process frame
            processed_frame, count = counter.process_frame(frame, car_icon)

            # Write frame to output video
            out.write(processed_frame)

        # Update session state
        st.session_state.total_count = counter.total_count

        return counter.total_count

    except Exception as e:
        st.error(f"Error processing video: {e}")
        return counter.total_count
    finally:
        # Release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()


# Main app
st.title("🚗 Car Counter App")
st.markdown("Upload a video to count the number of cars, trucks, buses, and motorcycles.")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")

    # Counting line settings
    st.subheader("Counting Line Position")
    line_x1 = st.slider("Line Start X", 0, 1000, st.session_state.counting_line[0])
    line_y1 = st.slider("Line Start Y", 0, 1000, st.session_state.counting_line[1])
    line_x2 = st.slider("Line End X", 0, 1000, st.session_state.counting_line[2])
    line_y2 = st.slider("Line End Y", 0, 1000, st.session_state.counting_line[3])

    st.session_state.counting_line = [line_x1, line_y1, line_x2, line_y2]

    # Performance settings
    st.subheader("Performance Settings")
    frame_skip = st.slider("Skip Frames", 0, 5, 0,
                           help="Skip frames for faster processing (0 = process all frames)")

    # Confidence threshold
    confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.3, 0.1)

    st.header("Upload Video")
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi", "mkv"])

    st.header("Car Icon")
    icon_option = st.radio("Select icon option:",
                           ["Use default icon", "Upload custom icon"])

    icon_path = "caricon.png"  # Default path

    if icon_option == "Upload custom icon":
        uploaded_icon = st.file_uploader("Upload a PNG icon", type=["png"])
        if uploaded_icon is not None:
            # Save uploaded icon to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_icon:
                tmp_icon.write(uploaded_icon.read())
                icon_path = tmp_icon.name

    if uploaded_file is not None:
        # Display video info
        file_details = {"FileName": uploaded_file.name, "FileSize": uploaded_file.size}
        st.info(f"Video uploaded successfully! {file_details['FileName']} ({file_details['FileSize']} bytes)")

        # Process button
        if st.button("Process Video", type="primary", use_container_width=True):
            st.session_state.processing = True

            # Save uploaded file to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_file.write(uploaded_file.read())
                input_path = tmp_file.name

            # Create output file
            output_path = "processed_video.mp4"

            # Process video
            with st.spinner("Processing video..."):
                count = process_video(input_path, output_path, icon_path,
                                      st.session_state.counting_line, frame_skip)
                st.session_state.processed_video = output_path

            # Clean up
            try:
                os.unlink(input_path)
                if icon_option == "Upload custom icon":
                    os.unlink(icon_path)
            except:
                pass

            st.session_state.processing = False
            st.rerun()

# Main content area
col1, col2 = st.columns(2)

with col1:
    st.header("Input Video")
    if uploaded_file is not None:
        st.video(uploaded_file)

        # Show counting line visualization
        st.subheader("Counting Line Preview")
        line_img = np.zeros((300, 600, 3), dtype=np.uint8)
        x1, y1, x2, y2 = st.session_state.counting_line
        # Scale coordinates to fit preview
        scale_x, scale_y = 600 / 1000, 300 / 1000
        px1, py1 = int(x1 * scale_x), int(y1 * scale_y)
        px2, py2 = int(x2 * scale_x), int(y2 * scale_y)
        cv2.line(line_img, (px1, py1), (px2, py2), (0, 0, 255), 2)
        st.image(line_img, caption="Counting Line Position", use_column_width=True)
    else:
        st.info("Please upload a video using the sidebar.")

with col2:
    st.header("Processed Video")
    if st.session_state.processed_video is not None and os.path.exists(st.session_state.processed_video):
        st.video(st.session_state.processed_video)
        st.success(f"Total vehicles counted: {st.session_state.total_count}")

        # Download button
        with open(st.session_state.processed_video, "rb") as file:
            btn = st.download_button(
                label="Download processed video",
                data=file,
                file_name="processed_video.mp4",
                mime="video/mp4",
                use_container_width=True
            )
    else:
        if st.session_state.processing:
            st.info("Video is being processed...")
            st.spinner("Processing")
        else:
            st.info("Processed video will appear here.")

# Add some information about the app
st.markdown("---")
st.markdown("""
### How it works:
1. Upload a video using the sidebar
2. Adjust the counting line position as needed
3. Optionally upload a custom car icon (PNG format)
4. Click the 'Process Video' button
5. The app will detect and count vehicles using YOLOv8
6. Download the processed video with bounding boxes, car icon, and count display

### Vehicle classes detected:
- Cars 🚗
- Trucks 🚚
- Buses 🚌
- Motorcycles 🏍️

### Tips for best results:
- Position the counting line where vehicles cross
- Use higher quality videos for better detection
- Adjust the confidence threshold if needed
- Use frame skipping for faster processing of long videos
""")

# Footer
st.markdown("---")

st.caption("Built with YOLOv8, SORT tracker, and Streamlit")

