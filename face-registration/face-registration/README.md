# Face Registration & Verification System

Real-time face registration and identity verification using live webcam, InsightFace, FastAPI, and MongoDB.

## Tech Stack

- **Backend** — Python, FastAPI, InsightFace (buffalo_l / ArcFace), OpenCV, asyncio
- **Database** — MongoDB (localhost)
- **Frontend** — HTML + Vanilla JS (live webcam, no file upload)

## How It Works

1. Student enters their name and captures 3 face photos via webcam (Front / Left / Right)
2. Backend extracts a **512-dimensional embedding** from each photo using InsightFace ArcFace model
3. All 3 embeddings are stored in MongoDB (1536 floats total per student)
4. To verify, capture one photo — backend compares it against all stored embeddings using **cosine similarity** and returns the best match with a confidence score

## Project Structure

```
face-registration-main/
├── src/
│   ├── app.py          # FastAPI app — /register and /verify endpoints
│   ├── config.py       # MongoDB connection (localhost:27017)
│   └── __init__.py
├── static/
│   └── index.html      # Frontend — live camera UI
├── requirements.txt
└── Dockerfile
```

## Setup & Run

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Start MongoDB** (must be running on port 27017)
```bash
# Windows
Start-Service MongoDB
```

**3. Start the server**
```bash
uvicorn src.app:app --host 0.0.0.0 --port 1010 --reload
```

**4. Open browser**
```
http://localhost:1010
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Serves the frontend |
| POST | `/register` | Register face — `name` + `front_image`, `left_image`, `right_image` |
| POST | `/verify` | Verify face — `verify_image` → returns matched name + confidence % |
| GET | `/students` | List all registered students |

## MongoDB

- **Database:** `face_registration_db`
- **Collection:** `face_embeddings`
- **Document structure:**
```json
{
  "name": "Student Name",
  "embeddings": {
    "front": [ 512 floats ],
    "left":  [ 512 floats ],
    "right": [ 512 floats ]
  },
  "registered_at": "2026-08-18T10:30:00"
}
```

## Requirements

```
fastapi
uvicorn[standard]
pymongo
opencv-python
numpy
insightface
onnxruntime
pydantic
python-multipart
```
