# src/ads/core/io.py
# -*- coding: utf-8 -*-
"""
ADSIO: Disk-first NIfTI IO bridge for nibabel, ANTsPy, and TorchIO,
with a minimal backend plugin registry for easy extensibility.

Principles
- Cross-ecosystem conversions *always* round-trip via a temporary NIfTI on disk.
- English-only logs/errors for stable automation.
- Clean functional core + a thin class façade (ADSIO) for reusable policy.
"""

from __future__ import annotations
import os
import json
import shutil
import tempfile
import numpy as np
from typing import Optional, Literal, Union, Tuple, Dict, Callable, NamedTuple

# ---------------------------
# Optional dependencies (checked at call sites)
# ---------------------------
try:
    import nibabel as nib
except Exception:
    nib = None

try:
    import ants  # antspyx
except Exception:
    ants = None

try:
    import torchio as tio
except Exception:
    tio = None

try:
    import torch
except Exception:
    torch = None


# ---------------------------
# Types
# ---------------------------
Backend = Literal["nib", "ants", "tio"]
AnyPath = Union[str, os.PathLike]
AnyObj = Union[object, AnyPath]


# ---------------------------
# Utils
# ---------------------------
from pathlib import Path
def load_nifti(path: Path) -> np.ndarray:
    """Load a NIfTI file and return a 3D numpy array."""
    return np.squeeze(nib.as_closest_canonical(nib.load(str(path))).get_fdata())

def _require(pkg, name: str):
    if pkg is None:
        raise ImportError(f"Required package '{name}' is not installed. Try: pip install {name}")

def _tmpdir(prefix="adsio-") -> tempfile.TemporaryDirectory:
    return tempfile.TemporaryDirectory(prefix=prefix)

def _set_descrip_safely(hdr: "nib.nifti1.Nifti1Header", text: str):
    """
    NIfTI 'descrip' is max 80 bytes. Truncate safely and set.
    """
    safe = text.encode("ascii", errors="ignore")[:79].decode("ascii", errors="ignore")
    hdr["descrip"] = safe

def _maybe_add_json_extension(img: "nib.Nifti1Image", obj: dict):
    """Attach JSON provenance as NIfTI extension (ecode=6)."""
    if nib is None:
        return img
    try:
        payload = json.dumps(obj, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        img.header.extensions.append(nib.nifti1.Nifti1Extension(6, payload))
    except Exception:
        pass
    return img

# ---------------------------
# Simple backend plugin registry (extensible)
# ---------------------------
class BackendOps(NamedTuple):
    load: Callable[[str], object]
    save: Callable[[object, str], None]
    detect: Callable[[object], bool]  # True if object belongs to this backend

_BACKENDS: Dict[str, BackendOps] = {}

def register_backend(name: str, ops: BackendOps):
    if name in _BACKENDS:
        raise ValueError(f"Backend '{name}' is already registered")
    _BACKENDS[name] = ops

def has_backend(name: str) -> bool:
    return name in _BACKENDS

def load_ext(path: str, backend_name: str):
    if backend_name not in _BACKENDS:
        raise ValueError(f"Unknown backend '{backend_name}'")
    return _BACKENDS[backend_name].load(path)

def save_ext(obj: object, path: str, backend_name: str):
    if backend_name not in _BACKENDS:
        raise ValueError(f"Unknown backend '{backend_name}'")
    return _BACKENDS[backend_name].save(obj, path)

def detect_ext(obj: object) -> Optional[str]:
    for name, ops in _BACKENDS.items():
        try:
            if ops.detect(obj):
                return name
        except Exception:
            pass
    return None


# ---------------------------
# In-memory backend detection
# ---------------------------
def detect_backend(obj: AnyObj) -> Optional[Backend]:
    """Detect backend of an in-memory object. Paths return None."""
    if isinstance(obj, (str, os.PathLike)):
        return None
    if nib is not None and isinstance(obj, getattr(nib, "spatialimages", object).SpatialImage):
        return "nib"
    if ants is not None and obj.__class__.__name__ == "ANTsImage":
        return "ants"
    if tio is not None and isinstance(obj, getattr(tio, "data", object).image.Image):
        return "tio"
    return None


# ---------------------------
# Heuristic on-disk provenance (best-effort)
# ---------------------------
def guess_nifti_writer(path: AnyPath) -> Dict[str, Union[str, float]]:
    """
    Heuristically guess which toolkit likely wrote this NIfTI.
    Returns:
      {"likely": "nibabel|ants|torchio|unknown", "confidence": 0..1, "notes": "..."}
    """
    _require(nib, "nibabel")
    p = str(path)
    info = {"likely": "unknown", "confidence": 0.0, "notes": "No distinctive header features."}
    try:
        img = nib.as_closest_canonical(nib.load(p))
        hdr = img.header
        has_sidecar = any(os.path.exists(os.path.splitext(p)[0] + ext)
                          for ext in (".json", ".yaml", ".yml"))
        has_ext = len(hdr.extensions) > 0
        qcode = int(hdr.get("qform_code", 0))
        scode = int(hdr.get("sform_code", 0))
        descrip = hdr.get("descrip", "").lower()
        notes = []
        if "torchio" in descrip:
            info["likely"] = "torchio"
            info["confidence"] = 0.9
            notes.append("Header 'descrip' contains 'torchio'.")
        elif has_sidecar:
            info["likely"] = "torchio"
            info["confidence"] = 0.7
            notes.append("Sidecar JSON/YAML found (common in TorchIO workflows).")
        elif not has_ext:
            info["likely"] = "ants"
            info["confidence"] = 0.6
            notes.append("No NIfTI extensions (often seen in ANTs-written files).")
        elif (qcode in (1, 2)) and (scode in (1, 2)):
            info["likely"] = "nibabel"
            info["confidence"] = 0.8
            notes.append(f"qform_code={qcode}, sform_code={scode} consistent with nib-style writes.")
        else:
            notes.append("No clear indicators for specific toolkit.")
        if notes:
            info["notes"] = " | ".join(notes)
        return info
    except Exception as e:
        return {"likely": "unknown", "confidence": 0.0, "notes": f"Failed to read header: {e}"}

# ---------------------------
# Low-level per-backend I/O (disk)
# ---------------------------
def _load_nib(path: AnyPath):
    _require(nib, "nibabel")
    return nib.as_closest_canonical(nib.load(str(path)))

def _load_nib_ras(path: AnyPath):
    _require(nib, "nibabel")
    img = nib.as_closest_canonical(nib.load(str(path)))
    return (img, True)

def _save_nib(obj, path: AnyPath):
    _require(nib, "nibabel")
    img = nib.as_closest_canonical(obj)
    nib.save(img, str(path))

def _load_ants(path: AnyPath):
    _require(ants, "ants")
    return ants.image_read(str(path))

def _save_ants(obj, path: AnyPath):
    _require(ants, "ants")
    ants.image_write(obj, str(path))

def _load_tio(path: AnyPath, *, as_label: bool = False):
    _require(tio, "torchio")
    return tio.LabelMap(str(path)) if as_label else tio.ScalarImage(str(path))

def _save_tio(obj, path: AnyPath):
    _require(tio, "torchio")
    if hasattr(obj, "save"):
        obj.save(str(path))
        return
    _require(torch, "torch")
    if isinstance(obj, np.ndarray):
        tensor = torch.as_tensor(obj, dtype=torch.float32)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
    elif isinstance(obj, torch.Tensor):
        tensor = obj.detach().cpu().float()
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
    else:
        raise TypeError("Unsupported object for TorchIO save.")
    tio.ScalarImage(tensor=tensor).save(str(path))


# Register built-ins into the plugin registry
if nib is not None:
    register_backend("nib", BackendOps(
        load=lambda p: _load_nib(p),
        save=lambda o, p: _save_nib(o, p),
        detect=lambda o: nib is not None and isinstance(o, getattr(nib, "spatialimages", object).SpatialImage),
    ))

if ants is not None:
    register_backend("ants", BackendOps(
        load=lambda p: _load_ants(p),
        save=lambda o, p: _save_ants(o, p),
        detect=lambda o: (ants is not None and o.__class__.__name__ == "ANTsImage"),
    ))

if tio is not None:
    register_backend("tio", BackendOps(
        load=lambda p: _load_tio(p, as_label=False),
        save=lambda o, p: _save_tio(o, p),
        detect=lambda o: (tio is not None and isinstance(o, getattr(tio, "data", object).image.Image)),
    ))


# ---------------------------
# Public, disk-first functional API
# ---------------------------
def load(path: AnyPath, backend: Backend, *, as_label: bool = False):
    if backend == "nib":
        return _load_nib(path)
    if backend == "ants":
        return _load_ants(path)
    if backend == "tio":
        return _load_tio(path, as_label=as_label)
    raise ValueError(f"Unknown backend: {backend}")

def save(obj: AnyObj, path: AnyPath, backend: Optional[Backend] = None, *,
         canonical: bool = False, as_label: bool = False) -> str:
    """
    Save object to disk. If backend is specified, convert to that backend first (disk-first).
    Returns final path (.nii.gz).
    """
    out_path = str(path)

    if backend is not None:
        obj = to_backend(obj, backend, canonical=canonical, as_label=as_label)

    obj_be = detect_backend(obj)
    if obj_be == "nib":
        _save_nib(obj, out_path, canonical=canonical)
    elif obj_be == "ants":
        _save_ants(obj, out_path)
    elif obj_be == "tio":
        _save_tio(obj, out_path)
    else:
        raise TypeError("Unable to infer object backend; pass a nib/ants/tio object or set 'backend'.")
    return out_path

def to_backend(obj: AnyObj, target: Backend, *,
               canonical: bool = False, as_label: bool = False):
    """
    Safe conversion to a target backend via temp NIfTI.
    - If obj is a path: load via target backend.
    - If in-memory: write as native backend -> read via target backend.
    """
    if isinstance(obj, (str, os.PathLike)):
        return load(obj, target, as_label=as_label)

    src = detect_backend(obj)
    if src is None:
        # Try plugin detection if available
        src = detect_ext(obj)  # may return "nib"/"ants"/"tio" or a plugin name
        if src is None:
            raise TypeError("Unsupported in-memory object. Provide a known backend object or a file path.")

    with _tmpdir() as tdir:
        mid = os.path.join(tdir, "bridge.nii.gz")
        if src in ("nib", "ants", "tio"):
            if src == "nib":
                _save_nib(obj, mid, canonical=canonical)
            elif src == "ants":
                _save_ants(obj, mid)
            elif src == "tio":
                _save_tio(obj, mid)
        else:
            # Plugin path: rely on plugin save to write NIfTI
            save_ext(obj, mid, src)
        return load(mid, target, as_label=as_label)

def convert_path(src_path: AnyPath, dst_path: AnyPath, dst_backend: Backend, *,
                 canonical: bool = False, as_label: bool = False) -> str:
    """
    Path-to-path conversion. Attempts nib->ants->tio (then plugins) for reading, then writes via dst_backend.
    """
    last_err = None
    obj = None
    # Try built-ins first
    for be in ("nib", "ants", "tio"):
        try:
            obj = load(src_path, be)
            break
        except Exception as e:
            last_err = e
    # Try plugins if built-ins failed
    if obj is None and _BACKENDS:
        for name in _BACKENDS.keys():
            try:
                obj = load_ext(str(src_path), name)
                break
            except Exception as e:
                last_err = e
    if obj is None:
        raise RuntimeError(f"Failed to read source with known backends. Last error: {last_err}")

    tgt = to_backend(obj, dst_backend, canonical=canonical, as_label=as_label)
    return save(tgt, dst_path, backend=dst_backend, canonical=canonical, as_label=as_label)

def roundtrip_identity(path: AnyPath, backend_a: Backend, backend_b: Backend,
                       *, canonical: bool = False) -> Tuple[str, str]:
    """
    A->B->A roundtrip via temp files; returns two persisted copies for external QC.
    """
    with _tmpdir(prefix="adsio-rt-") as tdir:
        a = load(path, backend_a)
        b = to_backend(a, backend_b, canonical=canonical)
        a2b = save(b, os.path.join(tdir, f"a2b_{backend_a}_to_{backend_b}.nii.gz"),
                   backend=backend_b, canonical=canonical)

        a_back = to_backend(b, backend_a, canonical=canonical)
        b2a = save(a_back, os.path.join(tdir, f"b2a_{backend_b}_to_{backend_a}.nii.gz"),
                   backend=backend_a, canonical=canonical)

        outdir = tempfile.mkdtemp(prefix="adsio-rt-out-")
        a2b_out = shutil.copy2(a2b, os.path.join(outdir, os.path.basename(a2b)))
        b2a_out = shutil.copy2(b2a, os.path.join(outdir, os.path.basename(b2a)))
        return a2b_out, b2a_out


# ---------------------------
# Helpers (provenance-aware creation, RAS loader)
# ---------------------------
def Nib_from_array_like_using_reference(data, reference) -> "nib.Nifti1Image":
    """
    Build a nib.Nifti1Image from a 3D/4D array/tensor using spatial metadata from reference
    (ANTsImage or TorchIO Image).
    """
    _require(nib, "nibabel"); _require(np, "numpy")
    if 'torch' in globals() and torch is not None and isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    if data is True or data is False:
        data = np.array(data, dtype=np.float32)
    if data.ndim == 4:
        data = np.squeeze(data)
    if data.ndim != 3:
        raise ValueError(f"Data must be 3D or 4D with a singleton channel; got shape {getattr(data,'shape',None)}")

    if ants is not None and reference.__class__.__name__ == "ANTsImage":
        tio_ref = ants_to_torchio_disk_first(reference)
        affine = tio_ref.affine
    elif tio is not None and isinstance(reference, getattr(tio, "data", object).image.Image):
        affine = reference.affine
    else:
        raise TypeError("Reference must be an ANTsImage or TorchIO Image.")

    return nib.Nifti1Image(np.asarray(data, dtype=np.float32), affine)

def nib_load_ras(path: AnyPath):
    _require(nib, "nibabel")
    return nib.as_closest_canonical(nib.load(str(path)))

def get_new_NibImgJ(new_img, temp_imgJ, dataType=np.float32):
    temp_imgJ.set_data_dtype(dataType)
    if new_img.dtype != np.dtype(dataType):
        new_img = new_img.astype(np.dtype(dataType))
    img_header = temp_imgJ.header.copy()
    img_header['glmax'] = np.max(new_img)
    img_header['glmin'] = np.min(new_img)
    img_header['xyzt_units'] = 0
    img_header['descrip'] = 'ADS Sept 2025'
    out = nib.Nifti1Image(new_img, temp_imgJ.affine, img_header)
    out.header.set_slope_inter(1, 0)
    return out

# ---------------------------
# Disk-first conversions between TorchIO/ANTs (helpers)
# ---------------------------
def torchio_to_ants_disk_first(img) -> "ants.ANTsImage":
    _require(tio, "torchio"); _require(ants, "ants")
    with _tmpdir(prefix="tio2ants-") as tdir:
        mid = os.path.join(tdir, "tio.nii.gz")
        _save_tio(img, mid)
        return _load_ants(mid)

def ants_to_torchio_disk_first(ants_image, *, as_label: bool = False):
    _require(ants, "ants"); _require(tio, "torchio")
    with _tmpdir(prefix="ants2tio-") as tdir:
        mid = os.path.join(tdir, "ants.nii.gz")
        _save_ants(ants_image, mid)
        return _load_tio(mid, as_label=as_label)

def ants_to_nib(ants_image) -> "nib.Nifti1Image":
    _require(ants, "ants"); _require(nib, "nibabel")
    with _tmpdir(prefix="ants2nib-") as tdir:
        mid = os.path.join(tdir, "ants.nii.gz")
        _save_ants(ants_image, mid)
        return _load_nib(mid)
    
def nib_to_ants(nib_image) -> "ants.ANTsImage":
    _require(ants, "ants"); _require(nib, "nibabel")
    with _tmpdir(prefix="nib2ants-") as tdir:
        mid = os.path.join(tdir, "nib.nii.gz")
        _save_nib(nib_image, mid)
        return _load_ants(mid)
    
# ---------------------------
# Backward-compatible shims (names you used earlier)
# ---------------------------
def torchio_to_ants(img):
    return torchio_to_ants_disk_first(img)

def ants_to_torchio(ants_image):
    return ants_to_torchio_disk_first(ants_image, as_label=False)

def save_nii_auto(data, path, reference=None):
    out = str(path)
    backend = detect_backend(data)
    if backend == "nib":
        return nib.save(nib.as_closest_canonical(data), out)
    elif backend == "ants":
        return ants.image_write(data, out)
    elif backend == "tio":
        return tio.ScalarImage(data).save(out)
    
    elif isinstance(data, (np.ndarray,)) or (torch is not None and isinstance(data, torch.Tensor)):
        if reference is None:
            raise ValueError("Saving raw arrays requires a reference image for spatial metadata.")
        # create nib image using reference, and get_new_NibImgJ function
        nib_img = get_new_NibImgJ(data, reference)
        return nib.save(nib.as_closest_canonical(nib_img), out)
    else:
        raise TypeError(f"Unsupported type for save_nii_auto: {type(data)}")

# ---------------------------
# ADSIO class façade (policy, reuse, orchestration)
# ---------------------------
class AdsIO:
    """
    A generalized neuroimaging I/O bridge supporting multiple backends.
    
    This class provides a unified interface for loading, saving, and converting
    neuroimaging data between different formats (nibabel, ANTsPy, TorchIO).
    It follows a "disk-first" approach for maximum compatibility.
    
    Args:
        canonical: Convert images to canonical orientation (RAS)
        as_label: Treat images as label maps when relevant
        read_order: Order of backends to try when auto-detecting formats
        persist_tmp: Keep temporary files for debugging
        tmp_prefix: Prefix for temporary directories
    """
    def __init__(self, *, canonical: bool = False, as_label: bool = False,
                 read_order=("nib", "ants", "tio"),
                 persist_tmp: bool = False, tmp_prefix: str = "adsio-"):
        self.canonical = canonical
        self.as_label = as_label
        self.read_order = tuple(read_order)
        self.persist_tmp = persist_tmp
        self._tmp = None
        self._tmp_prefix = tmp_prefix

    def __enter__(self):
        if self.persist_tmp:
            self._tmp = tempfile.TemporaryDirectory(prefix=f"persist-{self._tmp_prefix}")
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._tmp is not None:
            self._tmp.cleanup()

    # Core I/O operations
    def load(self, path: AnyPath, backend: Backend):
        """Load an image from disk using the specified backend."""
        return load(path, backend, as_label=self.as_label)

    def save(self, obj: AnyObj, path: AnyPath, backend: Optional[Backend] = None):
        """Save an object to disk, optionally converting to a specific backend first."""
        return save(obj, path, backend=backend, canonical=self.canonical, as_label=self.as_label)

    def to_backend(self, obj: AnyObj, target: Backend):
        """Convert an object to a specific backend format."""
        return to_backend(obj, target, canonical=self.canonical, as_label=self.as_label)
    
    def save_auto(self, data, path, reference=None):
        """Save data with automatic format detection."""
        out = str(path)
        backend = self.detect_backend(data)
        
        if backend is not None:
            return self.save(data, path, backend=backend)
        elif isinstance(data, (np.ndarray,)) or (torch is not None and isinstance(data, torch.Tensor)):
            if reference is None:
                raise ValueError("Saving raw arrays requires a reference image for spatial metadata.")
            nib_img = Nib_from_array_like_using_reference(data, reference)
            return self.save(nib_img, path, backend="nib")
        else:
            raise TypeError(f"Unsupported type for save_auto: {type(data)}")

    def convert_path(self, src: AnyPath, dst: AnyPath, dst_backend: Backend):
        """Convert an image file to another format using the configured read order."""
        last_err = None
        # Built-ins first
        for be in self.read_order:
            try:
                obj = load(src, be, as_label=self.as_label)
                break
            except Exception as e:
                last_err = e
        else:
            # Try plugins in order of registration
            obj = None
            for name in _BACKENDS.keys():
                try:
                    obj = load_ext(str(src), name)
                    break
                except Exception as e:
                    last_err = e
            if obj is None:
                raise RuntimeError(f"Failed to read with order {self.read_order} and plugins: {last_err}")

        tgt = self.to_backend(obj, dst_backend)
        return self.save(tgt, dst, backend=dst_backend)

    def roundtrip_identity(self, path: AnyPath, backend_a: Backend, backend_b: Backend,
                          *, canonical: Optional[bool] = None) -> Tuple[str, str]:
        """Perform an A→B→A roundtrip test, returning paths to the intermediate files."""
        can = self.canonical if canonical is None else canonical
        return roundtrip_identity(path, backend_a, backend_b, canonical=can)

    # Format conversion helpers
    def nib_to_ants(self, nib_image):
        """Convert a nibabel image to ANTs format."""
        return nib_to_ants(nib_image)
    
    def ants_to_nib(self, ants_image):
        """Convert an ANTs image to nibabel format."""
        return ants_to_nib(ants_image)
    
    def torchio_to_ants(self, tio_image):
        """Convert a TorchIO image to ANTs format."""
        return torchio_to_ants_disk_first(tio_image)
    
    def ants_to_torchio(self, ants_image):
        """Convert an ANTs image to TorchIO format."""
        return ants_to_torchio_disk_first(ants_image, as_label=self.as_label)

    # Stateless helpers
    @staticmethod
    def detect_backend(obj: AnyObj) -> Optional[Backend]:
        """Detect the backend of an in-memory object."""
        be = detect_backend(obj)
        return be if be is not None else detect_ext(obj)

    @staticmethod
    def guess_writer(path: AnyPath) -> Dict[str, Union[str, float]]:
        """Guess which toolkit wrote a NIfTI file."""
        return guess_nifti_writer(path)

    @staticmethod
    def nib_load_ras(path: AnyPath):
        """Load a NIfTI file in RAS orientation."""
        return nib_load_ras(path)

    @staticmethod
    def get_new_NibImgJ(new_img, temp_imgJ, dataType=np.float32):
        """Create a new nibabel image with reference metadata."""
        return get_new_NibImgJ(new_img, temp_imgJ, dataType=dataType)
# ----------------
