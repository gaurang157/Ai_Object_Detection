import requests
import streamlit as st
import av
import logging

logging.basicConfig(level=logging.WARNING)
st.set_page_config(page_title="Ai Object Detection", page_icon="🤖")
from PIL import Image
from ultralytics import YOLO
import cv2
import numpy as np
from io import BytesIO
from streamlit_webrtc import (
    ClientSettings,
    VideoTransformerBase,
    WebRtcMode,
    webrtc_streamer,
)
import supervision as sv
import numpy as np

# # Screen parameters
# screen_size_inch = 24  # Adjust this to your desired screen size
# aspect_ratio = (1, 1)  # Adjust this to your desired aspect ratio

# # Calculate diagonal size of the screen
# diagonal_size_cm = np.sqrt(screen_size_inch**2 + aspect_ratio[0]**2)

# # Calculate frame width and height
# frame_width = int((aspect_ratio[0] / np.sqrt(aspect_ratio[0]**2 + aspect_ratio[1]**2)) * diagonal_size_cm * 37.8)
# frame_height = int(frame_width * aspect_ratio[1] / aspect_ratio[0])
# print(f"fw->_{frame_width}_fh->_{frame_height}_")
# # Adjust the hyperparameters to change the size of the red zone
# red_zone_width_ratio = 0.4  # Adjust this to change the width of the red zone
# red_zone_height_ratio = 0.4  # Adjust this to change the height of the red zone

# # Calculate red zone dimensions
# red_zone_width = int(frame_width * red_zone_width_ratio)
# red_zone_height = int(frame_height * red_zone_height_ratio)

# # Calculate red zone position
# red_zone_x = (frame_width - red_zone_width) // 2
# red_zone_y = (frame_height - red_zone_height) // 2

# Define the zone polygon
zone_polygon_m = np.array([[160, 100], 
                         [160, 380], 
                         [481, 380], 
                         [481, 100]], dtype=np.int32)



# Initialize the YOLOv5 model
model = YOLO("yolov8n.pt")

# Initialize the tracker, annotators, and zone
tracker = sv.ByteTrack()
box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()
zone = sv.PolygonZone(polygon=zone_polygon_m, frame_resolution_wh=(642, 642))


zone_annotator = sv.PolygonZoneAnnotator(
    zone=zone,
    color=sv.Color.red(),
    thickness=2,
    text_thickness=4,
    text_scale=2
)

# def callback(frame: np.ndarray, _: int) -> np.ndarray:
#     # result = model(frame, agnostic_nms=True)[0]
#     results1 = model(frame)[0]
#     detections = sv.Detections.from_ultralytics(results1)
#     detections = tracker.update_with_detections(detections)
#     detections = detections[detections.confidence > 0.30]

#     labels = [
#         f"#{tracker_id} {results1.names[class_id]}"
#         for class_id, tracker_id
#         in zip(detections.class_id, detections.tracker_id)
#     ]

#     annotated_frame1 = box_annotator.annotate(frame.copy(), detections=detections)
#     annotated_frame1 = label_annotator.annotate(annotated_frame1, detections=detections, labels=labels)
#     zone.trigger(detections=detections)
#     frame1 = zone_annotator.annotate(scene=annotated_frame1)
#     return frame1


def main():
    st.title("Ai Object Detection")
    choice = st.radio("Select an option", ("Use webcam", "Cooking 🍳"))
    if choice == "Use webcam":
        # Define the WebRTC client settings
        client_settings = ClientSettings(
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
            media_stream_constraints={
                "video": True,
                "audio": False,
            },
        )

        # Define the WebRTC video transformer
        class ObjectDetector(VideoTransformerBase):
            def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
                # Convert the frame to an image
                img = Image.fromarray(frame.to_ndarray())
        
                # Run inference on 'bus.jpg' with arguments
                results = model.predict(img)
        
                # Ensure results is a valid object with necessary attributes
                # You might need to adjust this part based on the YOLO model you are using
                if isinstance(results, list):
                    results1 = results[0]  # Assuming the first element of the list contains the results
                else:
                    results1 = results
        
                detections = sv.Detections.from_ultralytics(results1)
                detections = tracker.update_with_detections(detections)
                detections = detections[detections.confidence > 0.30]
        
                labels = [
                    f"#{tracker_id} {results1.names[class_id]}"
                    for class_id, tracker_id in zip(detections.class_id, detections.tracker_id)
                ]
        
                # Convert av.VideoFrame to NumPy array
                frame_array = frame.to_ndarray(format="bgr24").copy()
        
                annotated_frame1 = box_annotator.annotate(frame_array, detections=detections)
                annotated_frame1 = label_annotator.annotate(annotated_frame1, detections=detections, labels=labels)
                zone.trigger(detections=detections)
                frame1 = zone_annotator.annotate(scene=annotated_frame1)
                # Display the count on the screen
                st.text(f"Objects in Zone: {zone.current_count}")
                # Convert the frame back to av.VideoFrame
                annotated_frame = av.VideoFrame.from_ndarray(frame1, format="bgr24")
                return annotated_frame


            

        # Start the WebRTC streamer
        webrtc_streamer(
            key="object-detection",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
            media_stream_constraints={
                "video": True,
                "audio": False,
            },
            video_processor_factory=ObjectDetector,
        )


if __name__ == '__main__':
    main()
