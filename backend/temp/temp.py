from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame
import cv2
import supervision as sv

# 1. Initialize annotators for visualization
annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# 2. Define what happens for every prediction
def render_prediction(predictions: dict, video_frame: VideoFrame):
    # Convert predictions to supervision Detections object
    detections = sv.Detections.from_inference(predictions)
    
    # Annotate the frame
    image = video_frame.image.copy()
    image = annotator.annotate(scene=image, detections=detections)
    image = label_annotator.annotate(scene=image, detections=detections)
    
    # Show the frame
    cv2.imshow("Weapon Detection", image)
    cv2.waitKey(1)

# 3. Initialize and start the pipeline
pipeline = InferencePipeline.init(
    model_id="weapon-detection-entlp/1", # The model from the link
    video_reference="gun4.mp4",    # Path to your local video file
    on_prediction=render_prediction,     # The callback function defined above
    api_key="YOUR_ROBOFLOW_API_KEY"      # Get this from your Roboflow dashboard
)

pipeline.start()
pipeline.join()