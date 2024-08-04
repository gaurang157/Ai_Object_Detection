import requests
import streamlit as st
import av
import logging
import os
import tempfile
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
import glob
import sys
import time
# Define the zone polygon
zone_polygon_m = np.array([[160, 100], 
						 [160, 380], 
						 [481, 380], 
						 [481, 100]], dtype=np.int32)
# # Calculate the center of the polygon
# center = np.mean(zone_polygon_m, axis=0)

# # Compute the vector from the center to each corner
# vectors = zone_polygon_m - center

# # Scale each vector by 20%
# expanded_vectors = vectors * 1.2

# # Added the expanded vectors to the center to get the new corners
# zone_polygon_m = (expanded_vectors + center).astype(np.int32)
@st.cache_resource
def load_yolo_model():
    return YOLO("yolov8n")

# Load the YOLO model (this will be cached)
model = load_yolo_model()

# Initialize the tracker, annotators and zone
tracker = sv.ByteTrack()
box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()
zone = sv.PolygonZone(polygon=zone_polygon_m, frame_resolution_wh=(642, 642))
mask_annotator = sv.MaskAnnotator()


zone_annotator = sv.PolygonZoneAnnotator(
	zone=zone,
	color=sv.Color.red(),
	thickness=2,
	text_thickness=4,
	text_scale=2
)

def draw_annotations(frame, boxes, masks, names):
	for box, name in zip(boxes, names):
		color = (0, 255, 0)  # Green color for bounding boxes

		# Draw bounding box
		cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)

		# Check if masks are available
		if masks is not None:
			mask = masks[frame_number]
			alpha = 0.3  # Transparency of masks

			# Draw mask
			frame[mask > 0] = frame[mask > 0] * (1 - alpha) + np.array(color) * alpha

		# Display class name
		cv2.putText(frame, name, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

	return frame

# Define the initial confidence threshold


def main():
	st.title("🤖 Ai Object Detection")
	st.subheader("YOLOv8 & Streamlit WebRTC Integration :)")
	st.sidebar.title("Select an option ⤵️")
	choice = st.sidebar.radio("", ("Live Webcam Predict", "Capture Image And Predict",":rainbow[Multiple Images Upload -]🖼️🖼️🖼️", "Upload Video"),
							captions = ["Live Count in Zone. :red[(Slow)]🐌", "Click and Detect. :orange[(Recommended)] :green[(Super Fast)]⚡⚡", "Upload & Process Multiple Images. :orange[(Recommended)] :green[(Fast)]⚡", "Upload Video & Predict 🏗️:orange[(Work in Progress)]📽️🎞️"], index = 1)
	conf = st.slider("Score threshold", 0.0, 1.0, 0.3, 0.05)
	if choice == "Live Webcam Predict":
		# conf = st.slider("Score threshold", 0.0, 1.0, 0.5, 0.05)

		# Define the WebRTC client settings
		client_settings = ClientSettings(
			# rtc_configuration={
			# 	"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
			# },
			media_stream_constraints={
				"video": True,
				"audio": False,
			},
		)

		# Define the WebRTC video transformer
		class ObjectDetector(VideoTransformerBase):
			def __init__(self):
				self.prev_time = time.time()
				self.fps = 0

			def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
				# Convert the frame to an image
				img = Image.fromarray(frame.to_ndarray())
				# img = frame.to_ndarray(format="bgr24").copy()
				# Flip the image horizontally
				# img = np.flip(img, axis=1)
				results = model.predict(img)

				# Processing time for the current frame
				curr_time = time.time()
				exec_time = curr_time - self.prev_time
				self.prev_time = curr_time
				# Calculate FPS
				self.fps = 1 / exec_time if exec_time != 0 else self.fps


				if isinstance(results, list):
					results1 = results[0]  
				else:
					results1 = results
				
				detections = sv.Detections.from_ultralytics(results1)

				detections = detections[detections.confidence > conf]

				labels = [
					f"{results1.names[class_id]}"
					for class_id in detections.class_id
				]

				# Convert av.VideoFrame to NumPy array
				frame_array = frame.to_ndarray(format="bgr24").copy()
				# frame_array = np.flip(frame_array, axis=1)
				annotated_frame1 = box_annotator.annotate(frame_array, detections=detections)
				annotated_frame1 = label_annotator.annotate(annotated_frame1, detections=detections, labels=labels)
				zone.trigger(detections=detections)
				frame1 = zone_annotator.annotate(scene=annotated_frame1)

				count_text = f"Objects in Zone: {zone.current_count}"
				cv2.putText(frame1, count_text, (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)

				# Overlay FPS on the frame
				fps_text = f"FPS: {self.fps:.2f}"
				cv2.putText(frame_array, fps_text, (10, 60), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)

				# Convert the frame back to av.VideoFrame
				annotated_frame = av.VideoFrame.from_ndarray(frame1, format="bgr24")
				return annotated_frame

		# Start the WebRTC streamer
		webrtc_streamer(
			key="object-detection",
			mode=WebRtcMode.SENDRECV,
			# rtc_configuration={
			#     "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
			# },
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

			results = model.predict(cv2_img)

			if isinstance(results, list):
				results1 = results[0]  
			else:
				results1 = results
	
			detections = sv.Detections.from_ultralytics(results1)
			detections = detections[detections.confidence > conf]
			labels = [
				f"#{index + 1}: {results1.names[class_id]}"
				for index, class_id in enumerate(detections.class_id)
			]

			labels1 = [
						f"#{index + 1}: {results1.names[class_id]} (Accuracy: {detections.confidence[index]:.2f})"
						for index, class_id in enumerate(detections.class_id)
					]

			annotated_frame1 = box_annotator.annotate(cv2_img, detections=detections)
			annotated_frame1 = label_annotator.annotate(annotated_frame1, detections=detections, labels=labels)
			# Display the count on the screen
			# count_text = f"Objects in Zone: {zone.current_count}"     # IMP
			count_text = f"Objects in Frame: {len(detections)}" 
			cv2.putText(annotated_frame1, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
			# Convert the frame back to av.VideoFrame
			annotated_frame = av.VideoFrame.from_ndarray(annotated_frame1, format="bgr24")
			st.image(annotated_frame.to_ndarray(), channels="BGR")
			# Assuming results is an instance of ultralytics.engine.results.Results
			st.write(':orange[ Info : ⤵️ ]')
			st.json(labels1)
			st.subheader("",divider='rainbow')

	elif choice == ":rainbow[Multiple Images Upload -]🖼️🖼️🖼️":
		uploaded_files = st.file_uploader("Choose a images", type=['png', 'jpg', 'webp', 'bmp'], accept_multiple_files=True)
		for uploaded_file in uploaded_files:
			bytes_data = uploaded_file.read()
			st.write("filename:", uploaded_file.name)
			bytes_data = uploaded_file.getvalue()
			cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

			results = model.predict(cv2_img)

			if isinstance(results, list):
				results1 = results[0]  
			else:
				results1 = results
	
			detections = sv.Detections.from_ultralytics(results1)
			detections = detections[detections.confidence > conf]
			labels = [
				f"#{index + 1}: {results1.names[class_id]}"
				for index, class_id in enumerate(detections.class_id)
			]

			labels1 = [
						f"#{index + 1}: {results1.names[class_id]} (Accuracy: {detections.confidence[index]:.2f})"
						for index, class_id in enumerate(detections.class_id)
					]
			
			annotated_frame1 = mask_annotator.annotate(cv2_img,detections=detections)
			annotated_frame1 = box_annotator.annotate(annotated_frame1, detections=detections)
			annotated_frame1 = label_annotator.annotate(annotated_frame1, detections=detections, labels=labels)
			
			# Display the count on the screen
			# count_text = f"Objects in Zone: {zone.current_count}"    
			count_text = f"Objects in Frame: {len(detections)}" 
			cv2.putText(annotated_frame1, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
			# Convert the frame back to av.VideoFrame
			annotated_frame = av.VideoFrame.from_ndarray(annotated_frame1, format="bgr24")
			# Display the annotated frame using st.image
			# annotated_frame = results1[0].plot()
			st.image(annotated_frame.to_ndarray(), channels="BGR")
			# st.image(annotated_frame, channels="BGR")
			st.write(':orange[ Info : ⤵️ ]')
			st.json(labels1)
			st.subheader("",divider='rainbow')
	elif choice == "Upload Video":
		# st.title("🏗️ Work in Progress 📽️ 🎞️")
		
		# Gaurang is Working on it...
		clip = st.file_uploader("Choose a video file", type=['mp4'])

		if clip:
			# Read the content of the video file
			video_content = clip.read()
			# Convert the video content to a bytes buffer
			video_buffer = BytesIO(video_content)
			st.video(video_buffer)

			with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
				temp_filename = temp_file.name
				temp_file.write(video_content)

			with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file1:
				temp_filename1 = temp_file1.name
				output_path = temp_filename1
				names = model.model.names
				cap = cv2.VideoCapture(temp_filename)
				w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
				# out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'VP90'), fps, (w, h))
				out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'vp09'), fps, (w, h))
				# out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (w, h))
				
				while True:
					ret, frame = cap.read()
					
					if not ret:
						break  # Break the loop if there are no more frames
					
					results = model.predict(frame, conf= conf)
					try:
						annotated_frame = results[0].plot(masks=True)
						# st.image(annotated_frame, caption='Frame', use_column_width=True)
					except:
						annotated_frame = results[0].plot(masks=False)
						# st.image(annotated_frame, caption='Frame', use_column_width=True)
					# st.image(annotated_frame, caption='Frame', use_column_width=True)
					out.write(annotated_frame)


				# Release VideoWriter object
				out.release()
				# Release VideoCapture object
				cap.release()

				st.video(output_path)
				st.success("Video processing completed.")
				st.write(output_path)
			

			# st.success("Video processing completed.")

	
	st.subheader("",divider='rainbow')
	st.write(':orange[ Classes : ⤵️ ]')
	cls_name = model.names
	cls_lst = list(cls_name.values())
	st.write(f':orange[{cls_lst}]')
if __name__ == '__main__':
	main()
