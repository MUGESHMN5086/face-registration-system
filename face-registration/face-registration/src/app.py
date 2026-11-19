from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import asyncio
import logging
import os

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from insightface.app import FaceAnalysis

from src.config import get_collection

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── InsightFace ───────────────────────────────────────────────────────────────
# det_size 320×320 is ~4x faster than 640×640 for close-up webcam shots
face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=0, det_size=(320, 320))

# ── Thread pool ───────────────────────────────────────────────────────────────
# CPU-bound face detection must run off the async event loop.
# max_workers=3 lets all 3 registration images process in parallel.
_executor = ThreadPoolExecutor(max_workers=3)


# ── Lifespan: clean up executor on shutdown ───────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield  # app is running
    _executor.shutdown(wait=False)
    logger.info("Thread pool executor shut down")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Face Registration API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "static")
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _decode_and_embed(raw_bytes: bytes, label: str) -> list:
    """
    CPU-bound: decode JPEG bytes → detect face → return 512-d ArcFace embedding.
    Runs inside the thread pool — never on the async event loop.
    """
    arr = np.frombuffer(raw_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not decode {label} image")
    faces = face_app.get(img)
    if not faces:
        raise ValueError(f"No face detected in {label} image")
    return faces[0].embedding.tolist()


async def extract_embedding(raw_bytes: bytes, label: str) -> list:
    """
    Async wrapper — offloads CPU-bound embedding work to the thread pool
    using asyncio.get_running_loop() (get_event_loop() is deprecated in 3.10+).
    """
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(_executor, _decode_and_embed, raw_bytes, label)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


def _upsert_student(doc: dict) -> None:
    """
    Blocking MongoDB write — runs in the thread pool so it does not block
    the async event loop.
    """
    get_collection().update_one(
        {"name": doc["name"]},
        {"$set": doc},
        upsert=True,
    )


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def index():
    html_path = os.path.join(os.path.dirname(__file__), "..", "static", "index.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return {"message": "Face Registration API is running"}


@app.get("/students")
def list_students():
    """Return all registered student names and registration timestamps."""
    docs = get_collection().find({}, {"_id": 0, "name": 1, "registered_at": 1})
    return {"students": list(docs)}


# ── 1. REGISTER ───────────────────────────────────────────────────────────────
@app.post("/register", status_code=201)
async def register(
    name: str = Form(...),
    front_image: UploadFile = File(...),
    left_image:  UploadFile = File(...),
    right_image: UploadFile = File(...),
):
    """
    Register a student's face.
    Accepts: name (form field) + front / left / right face images.
    Reads all 3 files concurrently, then extracts embeddings in parallel
    via a thread pool, and upserts the document into MongoDB.
    """
    name = name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Student name must not be empty")

    logger.info("Registering: %s", name)

    # Read all 3 uploaded files concurrently (I/O — stays on event loop)
    raw_front, raw_left, raw_right = await asyncio.gather(
        front_image.read(),
        left_image.read(),
        right_image.read(),
    )

    # Extract all 3 embeddings in parallel (CPU — offloaded to thread pool)
    emb_front, emb_left, emb_right = await asyncio.gather(
        extract_embedding(raw_front, "Front"),
        extract_embedding(raw_left,  "Left"),
        extract_embedding(raw_right, "Right"),
    )

    now = datetime.now(timezone.utc).isoformat()
    doc = {
        "name": name,
        "embeddings": {
            "front": emb_front,
            "left":  emb_left,
            "right": emb_right,
        },
        "registered_at": now,
        "updated_at":    now,
    }

    # MongoDB write is blocking — run in thread pool to keep event loop free
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(_executor, _upsert_student, doc)

    return {"status": "success", "message": f"'{name}' registered successfully"}


# ── 2. VERIFY ─────────────────────────────────────────────────────────────────
@app.post("/verify")
async def verify(
    verify_image: UploadFile = File(...),
):
    """
    Verify a face against all registered students.
    Computes cosine similarity between the query embedding and every stored
    embedding (front, left, right), averages the 3 scores per student, and
    returns the best match if it exceeds the threshold.
    """
    logger.info("Verifying face…")

    raw = await verify_image.read()
    emb_query = await extract_embedding(raw, "Verify")

    all_students = list(get_collection().find({}))
    if not all_students:
        raise HTTPException(status_code=404, detail="No students registered yet")

    query_vec  = np.array(emb_query)
    query_norm = np.linalg.norm(query_vec)

    best_name  = None
    best_score = -1.0

    for student in all_students:
        embeddings = student.get("embeddings", {})
        scores = []
        for angle_emb in embeddings.values():
            v     = np.array(angle_emb)
            denom = query_norm * np.linalg.norm(v)
            s     = float(np.dot(query_vec, v) / denom) if denom else 0.0
            scores.append(s)
        avg = sum(scores) / len(scores) if scores else 0.0
        if avg > best_score:
            best_score = avg
            best_name  = student["name"]

    THRESHOLD = 0.45
    if best_score >= THRESHOLD:
        return {
            "status":     "match",
            "name":       best_name,
            "confidence": round(best_score * 100, 2),
            "message":    f"Matched: {best_name} ({round(best_score * 100, 2)}%)",
        }
    return {
        "status":     "no_match",
        "name":       None,
        "confidence": round(best_score * 100, 2),
        "message":    "No matching student found",
    }


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=True)
