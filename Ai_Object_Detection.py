import requests
import streamlit as st
import av
import logging
import os

# Set the environment variable
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
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
# model = YOLO("best (3).pt")
model = YOLO("yolov8n")

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




def main():
    st.title("Ai Object Detection")

    choice = st.radio("Select an option", ("Live Webcam Predict", "Capture Image And Predict","Multiple Images Upload -🖼️🖼️🖼️"))
    if choice == "Live Webcam Predict":
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
                # print(f"RESULTS===>>__{results}")
                # Ensure results is a valid object with necessary attributes
                # You might need to adjust this part based on the YOLO model you are using
                if isinstance(results, list):
                    results1 = results[0]  # Assuming the first element of the list contains the results
                else:
                    results1 = results
        
                detections = sv.Detections.from_ultralytics(results1)
                detections = detections[detections.confidence > 0.90]
                # print(f"DETECTIONS--->_{detections}")

                labels = [
                    f"{results1.names[class_id]}"
                    for class_id in detections.class_id
                ]

                # Convert av.VideoFrame to NumPy array
                frame_array = frame.to_ndarray(format="bgr24").copy()

                annotated_frame1 = box_annotator.annotate(frame_array, detections=detections)
                annotated_frame1 = label_annotator.annotate(annotated_frame1, detections=detections, labels=labels)
                zone.trigger(detections=detections)
                frame1 = zone_annotator.annotate(scene=annotated_frame1)

                # Display the count on the screen
                # st.text(f"Objects in Zone: {zone.current_count}")
                # Inside the recv method of ObjectDetector
                # Display the count on the frame using cv2.putText
                count_text = f"🤖 Objects in Zone: {zone.current_count}"
                # count_text = "ai-object-detection.streamlit.app"
                cv2.putText(frame1, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

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
    elif choice == "Capture Image And Predict":
        img_file_buffer = st.camera_input("Take a picture")

        if img_file_buffer is not None:
            # To read image file buffer with OpenCV:
            bytes_data = img_file_buffer.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

            # Check the type of cv2_img:
            # Should output: <class 'numpy.ndarray'>
            results = model.predict(cv2_img)
            # print(f"RESULTS===>>__{results}")
            # Ensure results is a valid object with necessary attributes
            # You might need to adjust this part based on the YOLO model you are using
            if isinstance(results, list):
                results1 = results[0]  # Assuming the first element of the list contains the results
            else:
                results1 = results
    
            detections = sv.Detections.from_ultralytics(results1)
            detections = detections[detections.confidence > 0.20]
            # print(f"DETECTIONS--->_{detections}")

            labels = [
                f"{results1.names[class_id]}"
                for class_id in detections.class_id
            ]

            # Convert av.VideoFrame to NumPy array
            # frame_array = frame.to_ndarray(format="bgr24").copy()

            annotated_frame1 = box_annotator.annotate(cv2_img, detections=detections)
            annotated_frame1 = label_annotator.annotate(annotated_frame1, detections=detections, labels=labels)
            zone.trigger(detections=detections)
            frame1 = zone_annotator.annotate(scene=annotated_frame1)

            # Display the count on the screen
            # st.text(f"Objects in Zone: {zone.current_count}")
            # Inside the recv method of ObjectDetector
            # Display the count on the frame using cv2.putText
            count_text = f"🤖 Objects in Zone: {zone.current_count}"
            # count_text = "ai-object-detection.streamlit.app"
            cv2.putText(frame1, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Convert the frame back to av.VideoFrame
            annotated_frame = av.VideoFrame.from_ndarray(frame1, format="bgr24")
                
            # st.write(type(annotated_frame))
            # st.image(annotated_frame)
            # st.image(cv2.imshow(annotated_frame))
            # Display the annotated frame using st.image
            st.image(annotated_frame.to_ndarray(), channels="BGR")
            # Check the shape of cv2_img:
            # Should output shape: (height, width, channels)
            # st.write(annotated_frame.shape)
    elif choice == "Multiple Images Upload -🖼️🖼️🖼️":
        uploaded_files = st.file_uploader("Choose a images", type=['png', 'jpg'], accept_multiple_files=True)
        for uploaded_file in uploaded_files:
            bytes_data = uploaded_file.read()
            st.write("filename:", uploaded_file.name)
            bytes_data = uploaded_file.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            # Check the type of cv2_img:
            # Should output: <class 'numpy.ndarray'>
            results = model.predict(cv2_img)
            # print(f"RESULTS===>>__{results}")
            # Ensure results is a valid object with necessary attributes
            # You might need to adjust this part based on the YOLO model you are using
            if isinstance(results, list):
                results1 = results[0]  # Assuming the first element of the list contains the results
            else:
                results1 = results
    
            detections = sv.Detections.from_ultralytics(results1)
            detections = detections[detections.confidence > 0.10]
            # print(f"DETECTIONS--->_{detections}")

            labels = [
                f"{results1.names[class_id]}"
                for class_id in detections.class_id
            ]

            # Convert av.VideoFrame to NumPy array
            # frame_array = frame.to_ndarray(format="bgr24").copy()

            annotated_frame1 = box_annotator.annotate(cv2_img, detections=detections)
            annotated_frame1 = label_annotator.annotate(annotated_frame1, detections=detections, labels=labels)
            # print(f"-=-=->_{annotated_frame1}")
            g1 = 1
            # Display the count on the screen
            # st.text(f"Objects in Zone: {zone.current_count}")
            # Inside the recv method of ObjectDetector
            # Display the count on the frame using cv2.putText
            # count_text = f"Objects in Zone: {zone.current_count}"     # IMP
            # count_text = f"Objects in Zone: {g1}" 
            count_text = "🤖"
            cv2.putText(annotated_frame1, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Convert the frame back to av.VideoFrame
            annotated_frame = av.VideoFrame.from_ndarray(annotated_frame1, format="bgr24")
                
            # st.write(type(annotated_frame))
            # st.image(annotated_frame)
            # st.image(cv2.imshow(annotated_frame))
            # Display the annotated frame using st.image
            st.image(annotated_frame.to_ndarray(), channels="BGR")
            # Assuming results is an instance of ultralytics.engine.results.Results
            
            st.text(labels)
            

if __name__ == '__main__':
    main()

