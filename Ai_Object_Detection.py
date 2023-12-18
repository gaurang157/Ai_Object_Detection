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
    st.title("🤖 Ai Object Detection")

    choice = st.radio("Select an option", ("Live Webcam Predict", "Capture Image And Predict",":rainbow[Multiple Images Upload -]🖼️🖼️🖼️", "Upload Video"),
                            captions = ["Live Count in Zone.", "Click and Detect. :orange[(Recommended)]", "Upload & Process Multiple Images. :orange[(Recommended)]", "Upload Video & Predict"])
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
                # Ensure results is a valid object with necessary attributes
                # You might need to adjust this part based on the YOLO model you are using
                if isinstance(results, list):
                    results1 = results[0]  # Assuming the first element of the list contains the results
                else:
                    results1 = results
        
                detections = sv.Detections.from_ultralytics(results1)
                detections = detections[detections.confidence > 0.25]

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
                count_text = f"Objects in Zone: {zone.current_count}"
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
            # Ensure results is a valid object with necessary attributes
            # You might need to adjust this part based on the YOLO model you are using
            if isinstance(results, list):
                results1 = results[0]  # Assuming the first element of the list contains the results
            else:
                results1 = results
    
            detections = sv.Detections.from_ultralytics(results1)
            detections = detections[detections.confidence > 0.25]
            labels = [
                f"#{index + 1}: {results1.names[class_id]}"
                for index, class_id in enumerate(detections.class_id)
            ]

            labels1 = [
                        f"#{index + 1}: {results1.names[class_id]} (Accuracy: {detections.confidence[index]:.2f})"
                        for index, class_id in enumerate(detections.class_id)
                    ]
            # Convert av.VideoFrame to NumPy array
            # frame_array = frame.to_ndarray(format="bgr24").copy()
            annotated_frame1 = box_annotator.annotate(cv2_img, detections=detections)
            annotated_frame1 = label_annotator.annotate(annotated_frame1, detections=detections, labels=labels)
            # Display the count on the screen
            # st.text(f"Objects in Zone: {zone.current_count}")
            # Inside the recv method of ObjectDetector
            # Display the count on the frame using cv2.putText
            # count_text = f"Objects in Zone: {zone.current_count}"     # IMP
            count_text = f"Objects in Frame: {len(detections)}" 
            cv2.putText(annotated_frame1, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            # Convert the frame back to av.VideoFrame
            annotated_frame = av.VideoFrame.from_ndarray(annotated_frame1, format="bgr24")
            st.image(annotated_frame.to_ndarray(), channels="BGR")
            # Assuming results is an instance of ultralytics.engine.results.Results
            st.write(':orange[ Info : ⤵️ ]')
            st.json(labels1)
            st.subheader("",divider='rainbow')
            # Check the shape of cv2_img:
            # Should output shape: (height, width, channels)
            # st.write(annotated_frame.shape)
    elif choice == ":rainbow[Multiple Images Upload -]🖼️🖼️🖼️":
        uploaded_files = st.file_uploader("Choose a images", type=['png', 'jpg'], accept_multiple_files=True)
        for uploaded_file in uploaded_files:
            bytes_data = uploaded_file.read()
            st.write("filename:", uploaded_file.name)
            bytes_data = uploaded_file.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            # Check the type of cv2_img:
            # Should output: <class 'numpy.ndarray'>
            results = model.predict(cv2_img)
            # Ensure results is a valid object with necessary attributes
            # You might need to adjust this part based on the YOLO model you are using
            if isinstance(results, list):
                results1 = results[0]  # Assuming the first element of the list contains the results
            else:
                results1 = results
    
            detections = sv.Detections.from_ultralytics(results1)
            detections = detections[detections.confidence > 0.25]
            labels = [
                f"#{index + 1}: {results1.names[class_id]}"
                for index, class_id in enumerate(detections.class_id)
            ]

            labels1 = [
                        f"#{index + 1}: {results1.names[class_id]} (Accuracy: {detections.confidence[index]:.2f})"
                        for index, class_id in enumerate(detections.class_id)
                    ]

            # Convert av.VideoFrame to NumPy array
            # frame_array = frame.to_ndarray(format="bgr24").copy()
            annotated_frame1 = box_annotator.annotate(cv2_img, detections=detections)
            annotated_frame1 = label_annotator.annotate(annotated_frame1, detections=detections, labels=labels)
            # Display the count on the screen
            # st.text(f"Objects in Zone: {zone.current_count}")
            # Inside the recv method of ObjectDetector
            # Display the count on the frame using cv2.putText
            # count_text = f"Objects in Zone: {zone.current_count}"     # IMP
            count_text = f"Objects in Frame: {len(detections)}" 
            cv2.putText(annotated_frame1, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            # Convert the frame back to av.VideoFrame
            annotated_frame = av.VideoFrame.from_ndarray(annotated_frame1, format="bgr24")
            # Display the annotated frame using st.image
            st.image(annotated_frame.to_ndarray(), channels="BGR")
            # Assuming results is an instance of ultralytics.engine.results.Results
            st.write(':orange[ Info : ⤵️ ]')
            st.json(labels1)
            st.subheader("",divider='rainbow')
    elif choice == "Upload Video":
        st.title("🏗️Work in Progress📽️🎞️")
        '''# import tempfile
        # clip = st.file_uploader("Choose a video file", type=['mp4'])

        # if clip:
        #             # Read the content of the video file
        #     video_content = clip.read()
        #     # Convert the video content to a bytes buffer
        #     video_buffer = BytesIO(video_content)
        #     st.video(video_buffer)
        #     with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
        #         temp_filename = temp_file.name
        #         temp_file.write(clip.read())

                

        #         results = model(temp_filename,show = False, stream=False, save = False)
        #         # # Read the content of the video file
        #         # # video_content1 = results.read()
        #         # # Convert the video content to a bytes buffer
        #         # video_buffer1 = BytesIO(results)
        #         # st.video(video_buffer1)

        #     # Display the processed video
        #     # st.video(output_path)            
        #     # st.video(results)
            

        #     st.success("Video processing completed.")'''


    st.subheader("",divider='rainbow')
    st.write(':orange[ Classes : ⤵️ ]')
    cls_name = model.names
    cls_lst = list(cls_name.values())
    st.write(f':orange[{cls_lst}]')
if __name__ == '__main__':
    main()

