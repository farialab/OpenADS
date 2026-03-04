# server/app.py
from __future__ import annotations
import asyncio, os, shutil, zipfile, io, subprocess, sys
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# --- Constants ---
# Assuming this script is in a 'server/' directory, this resolves to the project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
print(f"Project root resolved to: {PROJECT_ROOT}")

# Define key directories
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
OUTPUT_ROOT = Path(os.environ.get("OPENADS_OUTPUT_ROOT", str(PROJECT_ROOT / "output")))
EXAMPLE_DATA_ROOT = PROJECT_ROOT / "assets" / "examples"
UPLOAD_ROOT = Path(os.environ.get("OPENADS_UPLOAD_ROOT", str(PROJECT_ROOT / "raw_upload")))

def _ensure_writable_dir(path: Path, fallback: Path) -> Path:
    """Create directory; fallback to /tmp when permission is denied."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        return path
    except PermissionError:
        fallback.mkdir(parents=True, exist_ok=True)
        print(f"Permission denied for {path}, using fallback: {fallback}")
        return fallback


# Ensure necessary directories exist (with safe fallback for non-root container users)
OUTPUT_ROOT = _ensure_writable_dir(OUTPUT_ROOT, Path("/tmp/openads_output"))
UPLOAD_ROOT = _ensure_writable_dir(UPLOAD_ROOT, Path("/tmp/openads_raw_upload"))


def _resolve_output_root(output_root: Optional[str]) -> Path:
    if output_root and output_root.strip():
        return Path(output_root).expanduser().resolve()
    return OUTPUT_ROOT.resolve()


app = FastAPI(title="OpenADS API")

# --- Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Endpoints ---

@app.get("/api/example-subjects")
async def get_example_subjects():
    """Scans the assets/examples directory for available subjects."""
    subjects = {"DWI": [], "PWI": []}
    if not EXAMPLE_DATA_ROOT.is_dir():
        print(f"Warning: Example data directory not found at {EXAMPLE_DATA_ROOT}")
        return subjects

    for modality in ["dwi", "pwi"]:
        modality_dir = EXAMPLE_DATA_ROOT / modality
        if modality_dir.is_dir():
            subjects[modality.upper()] = [p.name for p in modality_dir.iterdir() if p.is_dir()]
    return subjects

@app.post("/api/upload-subject")
async def upload_subject_data(
    modality: str = Form(...),
    subject_id: str = Form(...),
    files: List[UploadFile] = File(...),
):
    """Handles uploading individual raw data files for a subject."""
    if not subject_id.strip():
        raise HTTPException(status_code=400, detail="Subject ID is required.")

    modality_up = modality.upper()
    if modality_up not in {"DWI", "PWI"}:
        raise HTTPException(status_code=400, detail="Upload only supports DWI or PWI.")

    modality_lc = modality.lower()
    base_dir = UPLOAD_ROOT / modality_lc / subject_id
    base_dir.mkdir(parents=True, exist_ok=True)

    for f in files:
        filename = Path(f.filename).name
        dest_path = base_dir / filename
        with dest_path.open("wb") as out:
            shutil.copyfileobj(f.file, out)

    return {
        "subject_id": subject_id,
        "saved_root": str(base_dir.resolve()),
        "modality": modality.upper(),
        "message": f"Successfully uploaded {len(files)} files to {base_dir}",
    }

@app.post("/api/run")
async def run_pipeline(
    modality: str = Form(...),
    subject_id: Optional[str] = Form(None),
    raw_root: Optional[str] = Form(None),
    dwi_raw_root: Optional[str] = Form(None),
    pwi_raw_root: Optional[str] = Form(None),
    stages: Optional[str] = Form(None),
    dwi_stages: Optional[str] = Form(None),
    pwi_stages: Optional[str] = Form(None),
    gpu: Optional[str] = Form(None),
    config: Optional[str] = Form(None),
    dwi_config: Optional[str] = Form(None),
    pwi_config: Optional[str] = Form(None),
    output_root: Optional[str] = Form(None),
):
    """Executes the pipeline script and streams output back to the client."""
    modality_up = modality.upper()
    cmd: list[str] = []

    if modality_up in {"DWI", "PWI"}:
        if not raw_root and subject_id:
            modality_lc = modality.lower()
            if (UPLOAD_ROOT / modality_lc / subject_id).is_dir():
                raw_root = str(UPLOAD_ROOT / modality_lc / subject_id)
            elif (EXAMPLE_DATA_ROOT / modality_lc / subject_id).is_dir():
                raw_root = str(EXAMPLE_DATA_ROOT / modality_lc / subject_id)
        if not raw_root:
            raise HTTPException(status_code=400, detail="Raw data folder not found.")

        script_name = "run_ads_dwi.py" if modality_up == "DWI" else "run_ads_pwi.py"
        cmd = [sys.executable, str(SCRIPTS_ROOT / script_name), "--subject-path", raw_root]
        if config:
            cmd.extend(["--config", config])
        if stages:
            cmd.extend(["--stages", stages])
        if gpu:
            cmd.extend(["--gpu", gpu])
        if output_root:
            cmd.extend(["--output-root", output_root])

    elif modality_up in {"DWI&PWI", "DWI_PWI", "COMBINED"}:
        dwi_root = dwi_raw_root
        pwi_root = pwi_raw_root
        if subject_id:
            if not dwi_root:
                if (UPLOAD_ROOT / "dwi" / subject_id).is_dir():
                    dwi_root = str(UPLOAD_ROOT / "dwi" / subject_id)
                elif (EXAMPLE_DATA_ROOT / "dwi" / subject_id).is_dir():
                    dwi_root = str(EXAMPLE_DATA_ROOT / "dwi" / subject_id)
            if not pwi_root:
                if (UPLOAD_ROOT / "pwi" / subject_id).is_dir():
                    pwi_root = str(UPLOAD_ROOT / "pwi" / subject_id)
                elif (EXAMPLE_DATA_ROOT / "pwi" / subject_id).is_dir():
                    pwi_root = str(EXAMPLE_DATA_ROOT / "pwi" / subject_id)

        if not dwi_root or not pwi_root:
            raise HTTPException(
                status_code=400,
                detail="Combined run requires both DWI and PWI subject folders.",
            )
        cmd = [
            sys.executable,
            str(SCRIPTS_ROOT / "run_ads_combined.py"),
            "--dwi-subject-path",
            dwi_root,
            "--pwi-subject-path",
            pwi_root,
        ]
        if dwi_config:
            cmd.extend(["--dwi-config", dwi_config])
        if pwi_config:
            cmd.extend(["--pwi-config", pwi_config])
        if dwi_stages:
            cmd.extend(["--dwi-stages", dwi_stages])
        if pwi_stages:
            cmd.extend(["--pwi-stages", pwi_stages])
        if gpu:
            cmd.extend(["--gpu", gpu])
        if output_root:
            cmd.extend(["--output-root", output_root])
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported modality: {modality}")

    async def _stream_subprocess():
        env = os.environ.copy()
        if gpu:
            env["CUDA_VISIBLE_DEVICES"] = "" if gpu in {"-1", "cpu", "none"} else gpu
        
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, env=env
        )
        yield f"data: Running command: {' '.join(cmd)}\n\n"
        while True:
            line = await proc.stdout.readline()
            if not line: break
            yield f"data: {line.decode('utf-8').rstrip()}\n\n"
        
        code = await proc.wait()
        yield f"data: __EXIT__:{code}\n\n"

    return StreamingResponse(_stream_subprocess(), media_type="text/event-stream")

@app.get("/api/outputs")
async def list_outputs(modality: str, subject_id: str, output_root: Optional[str] = None):
    """Lists all generated output files for a given subject."""
    modality_up = modality.upper()
    out_root = _resolve_output_root(output_root)
    subj_root = out_root / subject_id
    if modality_up in {"DWI", "PWI"}:
        modality_root = subj_root / modality_up
        search_dirs = [modality_root]
        rel_base = modality_root
    elif modality_up in {"DWI&PWI", "DWI_PWI", "COMBINED"}:
        search_dirs = [subj_root / "DWI", subj_root / "PWI"]
        rel_base = subj_root
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported modality: {modality}")

    files: List[str] = []
    for d in search_dirs:
        if not d.exists():
            continue
        files.extend(str(p.relative_to(rel_base)) for p in d.rglob("*") if p.is_file())
    if not files:
        raise HTTPException(status_code=404, detail="Output directory for subject not found.")

    return {"files": files}

@app.get("/api/download")
async def download_file(modality: str, subject_id: str, relpath: str, output_root: Optional[str] = None):
    """Downloads a single output file. Note: This relpath is relative to the subject dir."""
    out_root = _resolve_output_root(output_root)
    subj_root = (out_root / subject_id).resolve()
    modality_up = modality.upper()
    if modality_up in {"DWI", "PWI"}:
        allowed_roots = [(subj_root / modality_up).resolve()]
    elif modality_up in {"DWI&PWI", "DWI_PWI", "COMBINED"}:
        allowed_roots = [subj_root]
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported modality: {modality}")

    file_path = None
    for root in allowed_roots:
        candidate = (root / relpath).resolve()
        if str(candidate).startswith(str(root)) and candidate.exists():
            file_path = candidate
            break
    if file_path is None:
        raise HTTPException(status_code=404, detail="File not found or access denied.")
    
    return FileResponse(str(file_path), filename=file_path.name)

@app.get("/api/download-all")
async def download_all_outputs(modality: str, subject_id: str, output_root: Optional[str] = None):
    """Zips and downloads all output files for a subject."""
    out_root = _resolve_output_root(output_root)
    modality_up = modality.upper()
    subj_root = out_root / subject_id
    if modality_up in {"DWI", "PWI"}:
        roots = [subj_root / modality_up]
    elif modality_up in {"DWI&PWI", "DWI_PWI", "COMBINED"}:
        roots = [subj_root / "DWI", subj_root / "PWI"]
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported modality: {modality}")

    existing_roots = [r for r in roots if r.exists()]
    if not existing_roots:
        raise HTTPException(status_code=404, detail="Subject output directory not found.")

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for root in existing_roots:
            for p in root.rglob("*"):
                if p.is_file():
                    zf.write(p, arcname=str(p.relative_to(subj_root)))
    
    zip_buffer.seek(0)
    return StreamingResponse(
        zip_buffer, media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{subject_id}_outputs.zip"'}
    )

# --- Static File Serving ---
# KEY CHANGE: This exposes the entire project folder under the "/files" URL.
# This allows NiiVue to directly fetch files from `assets/` and `output/`.
app.mount("/files", StaticFiles(directory=str(PROJECT_ROOT)), name="files")

# This serves the main web page (index.html) from the 'web' directory.
app.mount("/", StaticFiles(directory=str(PROJECT_ROOT / "web"), html=True), name="web")
