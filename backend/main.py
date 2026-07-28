from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import shutil
from pathlib import Path
import subprocess
import os
import sys

# Ensure backend can be imported
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.core.generator import query_rag
from backend.core.config import RAW_DOCS_DIR

app = FastAPI(
    title="VDA-5050 Fleet Communication Oracle",
    description="REST API for the RAG Assistant",
    version="1.0.0"
)

class QueryRequest(BaseModel):
    question: str

@app.post("/query")
def query_endpoint(request: QueryRequest):

    try:
        result = query_rag(request.question)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest")
def ingest_endpoint(file: UploadFile = File(...)):

    try:
        RAW_DOCS_DIR.mkdir(parents=True, exist_ok=True)
        file_path = RAW_DOCS_DIR / file.filename
        
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Trigger the ingestion script to re-embed everything including the new file
        env = os.environ.copy()
        env["PYTHONPATH"] = str(_PROJECT_ROOT)
        
        result = subprocess.run(
            [sys.executable, "-m", "backend.core.run_ingestion"], 
            cwd=str(_PROJECT_ROOT),
            env=env,
            capture_output=True, 
            text=True
        )
        
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Ingestion failed: {result.stderr}")
            
        return {
            "message": f"Successfully uploaded {file.filename} and ran ingestion.", 
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
