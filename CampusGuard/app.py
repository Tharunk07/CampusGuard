import logging
import tempfile
import streamlit as st
import cv2
import threading
import os
import time
from ultralytics import YOLO
from telegram_sender import send_msg
from moviepy.editor import ImageSequenceClip


count = 1
detection_start_time = None
continuous_detection_threshold = 2  # 5 seconds of continuous detection
output_folder = "CapturedFrames"

original_log_level = logging.getLogger().getEffectiveLevel()

# Set the log level to a higher level, e.g., WARNING or CRITICAL
logging.disable(logging.CRITICAL)

def video_to_bytes(video_path):
    with open(video_path, 'rb') as video_file:
        video_bytes = video_file.read()
    return video_bytes


# Function to perform YOLO detection
def perform_detection(frame):
    global detection_start_time, count
    # Perform prediction on the frame
    results = model(frame)[0]

    # Draw bounding boxes and labels for each detection
    for box in results:
        # Get individual coordinates
        x1, y1, x2, y2 = box.boxes.xyxy[0].cpu().numpy()

        # Get class label and confidence score
        class_id = box.boxes.cls[0].cpu().numpy().item()
        label = d[class_id]  # Map ID to label
        conf = box.boxes.conf[0].cpu().numpy().item()
        color = colors[class_id]

        # Draw bounding box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        # Add label with optional confidence score
        text = f"{label}: {conf:.2f}"  # Format confidence score

        # Calculate label placement (adjust as needed)
        offset = 5  # Adjust offset based on text size and box size
        text_width, text_height = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        x_text = int(x1 + offset)
        y_text = int(y1 + offset + text_height)

        cv2.putText(frame, text, (x_text, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        if conf > 0.3:
            if detection_start_time is None:
                detection_start_time = time.time()
            else:
                elapsed_time = time.time() - detection_start_time
                if elapsed_time >= continuous_detection_threshold:
                    # print(f"Continuous detection for {continuous_detection_threshold} seconds!")
                    detection_start_time = None

                    # Save the captured frame
                    image_filename = os.path.join(output_folder, f"img_{count}.jpg")
                    cv2.imwrite(image_filename, frame)
                    count += 1

                    threading.Thread(target=lambda: send_msg(image_filename, label)).start()
# Streamlit app
# Streamlit app
# Streamlit app
# Initialize session state
if "cap" not in st.session_state:
    st.session_state.cap = None

print(st.session_state)
# Initialize frames in session state
if "frames" not in st.session_state:
    st.session_state.frames = []

def main():
    st.title("YOLOv8 Video Streamlit App")
    cap = None

    # Choose between uploading a video file and using the device camera
    option = st.radio("Choose an option:", ("Upload Video File", "Use Device Camera"))
    temp_dir = tempfile.TemporaryDirectory()
    if option == "Upload Video File":
        # Upload video file
        uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mkv"])

        if uploaded_file is not None:
            # Use a temporary directory to store the uploaded video
            temp_path = os.path.join(temp_dir.name, "uploaded_video.mp4")

            # Save the uploaded file to the temporary directory
            with open(temp_path, "wb") as video_file:
                video_file.write(uploaded_file.read())

            cap = cv2.VideoCapture(temp_path)
    elif option == "Use Device Camera":
        # Use the device camera
        cap = cv2.VideoCapture(0)  # 0 corresponds to the default camera, you can change it if needed
    else:
        st.error("Invalid option selected.")
        st.stop()

    if cap and not cap.isOpened():
        st.error("Error opening video/camera.")
        st.stop()

    if cap and cap.isOpened():
        image_container = st.empty()
        stop_button = st.button("Stop")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        timestamp = time.strftime("%Y%m%d%H%M%S")

        # Process each frame of the video
        # frames = []
        while True:
            # Capture a frame
            ret, frame = cap.read()

            # Break if frame is not captured
            if not ret:
                break

            # Perform YOLO detection on the frame (Placeholder for your YOLO detection function)
            # Uncomment the following line when YOLO detection is implemented
            perform_detection(frame)

            # Resize the frame to fit within the screen while maintaining the aspect ratio
            max_width = 800  # Adjust this value based on your layout
            aspect_ratio = frame_width / frame_height
            new_width = min(frame_width, max_width)
            new_height = int(new_width / aspect_ratio)
            resized_frame = cv2.resize(frame, (new_width, new_height))

            # Convert BGR to RGB before appending to the list
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

            # Append the RGB frame to the list
            st.session_state.frames.append(rgb_frame)

            # Display the frame with detections
            image_container.image(rgb_frame, channels="RGB", use_column_width=True)

            # Exit on button click
            if stop_button:
                break

        # Release resources
        cap.release()

        # Create an ImageSequenceClip from the frames list
        video_clip = ImageSequenceClip(st.session_state.frames, fps=fps)

        # Write the ImageSequenceClip to the output video file
        output_video_path = os.path.join(temp_dir.name, f"output_video_{timestamp}.mp4")
        video_clip.write_videofile(output_video_path, codec="libx264", audio=False)

        # Display download link for the output video
        st.success("Video processing completed!")

        # Provide the correct path for download
        st.download_button("Download Processed Video", data=video_to_bytes(output_video_path),
                        file_name=f"output_video_{timestamp}.mp4")



# Load the YOLO model
model = YOLO("best1.pt")

# Define the dictionary mapping class IDs to labels
d = {
    0: "Mobile Phone",  # Example mapping, update with your classes
    1: "No Helmet",
    2: "Sleeping",
    3: "Triples",
    4: "Violence"
    # ... other class ID mappings
}

colors = {
    0: (134, 34, 255),
    1: (254, 0, 86),
    2: (0, 255, 206),
    3: (255, 128, 0),
    4: (252, 169, 3)
}



# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Run the Streamlit app
if __name__ == "__main__":
    main()
