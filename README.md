# SurveilX-FYP

SurveilX is a project for monitoring video feeds with restricted area detection. It consists of a Flask backend and a React frontend. The system detects restricted area breaches in video streams.

## Project Structure
```
project-folder/
│
├─ backend/           # Flask backend
│  ├─ main.py         # Main backend script
│  ├─ pd.py           # YOLO integration and detection logic
│  ├─ vid1_v2/        # Sample videos (place your .mp4 files here)
│  ├─ yolov8m.pt      # YOLO model files (not included in repo)
│  ├─ yolov8n.pt
│  ├─ yolov8s.pt
│  └─ requirements.txt
│
├─ frontend/          # React frontend
│  ├─ public/
│  ├─ src/
│  ├─ package.json
│  └─ README
│
└─ .gitignore
```


---

## Setup Instructions

### Backend

1. Create a virtual environment:
```bash
cd backend
python -m venv venv
venv\Scripts\activate    # Windows
source venv/bin/activate # Linux/macOS
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. YOLO model files

The backend requires YOLO model files (yolov8m.pt, yolov8n.pt, yolov8s.pt) to work. These files are not included in the repo due to size.
You can download them from the official YOLO repository or from your provided source and place them inside the backend folder.

4. Video files
Place your .mp4 video in the backend/vid1_v2. The system will process these videos.

5. Run the backend:
```bash
uvicorn main:app --reload
```


### Frontend

1. Install dependencies
```bash
cd frontend
npm install
```

2. Start the frontend
```bash
npm start
```

- The frontend is available at http://localhost:3000 by default.

- It communicates with the Flask backend for video processing and restricted area detection.

## Notes

- Ensure the backend is running before starting the frontend.

- Make sure the YOLO model files and video files are placed correctly in the backend folder.

## Contact 

Mirza Amir Akbar Khan<br>
email: i221112@nu.edu.pk