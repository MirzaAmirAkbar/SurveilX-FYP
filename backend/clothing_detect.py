"""
SurveilX Clothing Attribute Detector
=====================================
Detects people in a video and labels each with: Hat, Glasses, LongSleeve.
"""

import os
import sys
import cv2
import torch
import numpy as np
import torchvision.models as models
import torchvision.transforms as T
from ultralytics import YOLO

# ── Config ────────────────────────────────────────────────────────────────────
# Corrected __file__ to get the directory where this script lives
_DIR        = os.path.dirname(os.path.abspath(__file__))

# Define paths to the video and the two AI models based on the current directory
VIDEO_PATH  = os.path.join(_DIR, "backend", "vid2.mp4")
HUMAN_MODEL = os.path.join(_DIR, "human_detection.pt")
CLOTH_MODEL = os.path.join(_DIR, "surveilx_clothing_classifier_FINETUNED.pth")

# The confidence score required to say "Yes, this person is wearing a hat/glasses"
THRESHOLD   = 0.50

# Automatically use the GPU if available to make it run faster, otherwise use the CPU
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# The master list of all attributes the MobileNetV3 model was trained to recognize
ATTR_NAMES = [
    'Female', 'AgeOver60', 'Age18-60', 'AgeLess18', 'Front', 'Side', 'Back',
    'Hat', 'Glasses', 'HandBag', 'ShoulderBag', 'Backpack', 'HoldObjectsInFront',
    'ShortSleeve', 'LongSleeve', 'UpperStride', 'UpperLogo', 'UpperPlaid',
    'UpperSplice', 'LowerStripe', 'LowerPattern', 'LongCoat', 'Trousers',
    'Shorts', 'Skirt&Dress', 'Boots'
]

# We only care about tracking these 3 specific items out of the master list above
_WANTED = {
    "Hat":        ATTR_NAMES.index('Hat'),
    "Glasses":    ATTR_NAMES.index('Glasses'),
    "LongSleeve": ATTR_NAMES.index('LongSleeve'),
}

# ── Reusable class (for integration) ─────────────────────────────────────────
class ClothingDetector:
    """
    This class handles the MobileNetV3 model. We put it in a class so you 
    could theoretically import this into another project easily.
    """

    # Image transformations: The AI needs images to be a specific size and format
    _transform = T.Compose([
        T.ToPILImage(),
        T.Resize((256, 128)),                                                 # Resize to 256x128 pixels
        T.ToTensor(),                                                         # Convert to a PyTorch tensor
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),   # Standard color normalization
    ])

    # Corrected __init__. This runs once when you create the ClothingDetector.
    def __init__(self, model_path: str = CLOTH_MODEL, threshold: float = THRESHOLD):
        self.threshold = threshold
        
        # Load the blank MobileNetV3 blueprint
        net = models.mobilenet_v3_large(weights=None)
        
        # Modify the last layer so it outputs the correct number of attributes (26)
        net.classifier[-1] = torch.nn.Linear(1280, len(ATTR_NAMES))
        
        # Load the custom "weights" (the actual trained "brain" of the model)
        net.load_state_dict(torch.load(model_path, map_location=DEVICE))
        
        # Move the model to the GPU/CPU and set it to evaluation (prediction) mode
        self.net = net.to(DEVICE).eval()

    def predict(self, crop_bgr: np.ndarray) -> dict:
        """Takes a cropped image of a person and returns a dictionary of what they are wearing."""
        
        # Convert image from BGR (OpenCV format) to RGB (Standard format)
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        
        # Apply our size/color transformations from above
        t = self._transform(rgb).unsqueeze(0).to(DEVICE)
        
        # Tell PyTorch we don't need to train, just predict (saves memory/speed)
        with torch.no_grad():
            probs = torch.sigmoid(self.net(t)[0]).cpu().numpy()
            
        # Return a True/False dictionary for our _WANTED items if confidence > THRESHOLD
        return {name: bool(probs[idx] >= self.threshold) for name, idx in _WANTED.items()}


# ── Drawing helper ────────────────────────────────────────────────────────────
def _draw(frame, x1, y1, x2, y2, attrs: dict):
    """Draws a bounding box and text labels onto the video frame."""
    
    # Draw a yellow box around the person
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
    
    # Figure out which attributes are True (e.g., if Hat and Glasses are True)
    active = [k for k, v in attrs.items() if v]
    label  = ", ".join(active) if active else "—" # Join them with commas, or use a dash if none
    
    # Calculate text size to draw a background box for the text to make it readable
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    pad = 4
    cy1 = max(y1 - th - pad * 2, 0)
    
    # Draw the solid background box for the text
    cv2.rectangle(frame, (x1, cy1), (x1 + tw + pad * 2, y1), (0, 200, 255), -1)
    
    # Draw the black text over the yellow background box
    cv2.putText(frame, label, (x1 + pad, y1 - pad),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)


# ── Main (standalone run) ─────────────────────────────────────────────────────
# Corrected __name__ and __main__. This ensures the code only runs if executed directly.
if __name__ == "__main__":
    print(f"[*] Device      : {DEVICE}")
    print(f"[*] Video path  : {VIDEO_PATH}")
    print(f"[*] File exists : {os.path.exists(VIDEO_PATH)}")

    print("[*] Loading human detector (YOLO) ...")
    yolo = YOLO(HUMAN_MODEL) # Load the YOLO model
    yolo.to(DEVICE)

    print("[*] Loading clothing classifier ...")
    clothing = ClothingDetector() # Initialize our custom class

    print("[*] Opening video file ...")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[!] Cannot open video: {VIDEO_PATH}. Check the filepath!")
        sys.exit(1)

    print("[*] Running — press Q on the video window to quit")

    # Start an infinite loop to process the video frame by frame
    while True:
        ret, frame = cap.read() # Grab the next frame
        
        # If the video ends, loop it back to the beginning
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # 1. Ask YOLO to find humans (class 0 is 'person' in standard YOLO models)
        results = yolo(frame, classes=[0], conf=0.40, verbose=False)
        boxes   = results[0].boxes # Get the bounding boxes of found humans

        # If we found at least one person...
        if boxes is not None and len(boxes):
            # 2. Loop through every single person found in the frame
            for (x1, y1, x2, y2) in boxes.xyxy.cpu().numpy().astype(int):
                
                # Make sure the box coordinates don't go outside the video boundaries
                x1, y1 = max(x1, 0), max(y1, 0)
                x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])
                
                # 3. Crop the image down to just this specific person
                crop = frame[y1:y2, x1:x2]
                
                # Skip if the crop somehow ended up with zero width or height
                if crop.size == 0:
                    continue
                    
                # 4. Predict the clothing attributes for this person
                attrs = clothing.predict(crop)
                
                # 5. Draw the results onto the main frame
                _draw(frame, x1, y1, x2, y2, attrs)

        # Show the processed frame in a pop-up window
        cv2.imshow("SurveilX - Clothing Detection", frame)
        
        # Wait 1 millisecond for user input. If they press 'q', break the loop.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up and close everything when done
    cap.release()
    cv2.destroyAllWindows()