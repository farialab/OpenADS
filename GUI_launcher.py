#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ADS Unified GUI (Shell Mode) v4.3
- Proper stage inputs/outputs from pipeline JSON
- Working output file display
- Smaller terminal (1/4 size)
- Auto-refresh after stage completion
"""

import sys
import os
import shutil
import subprocess
import csv
from html import escape
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable

from PyQt5.QtCore import Qt, QSettings, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox,
    QPushButton, QTextEdit, QVBoxLayout, QWidget, QProgressBar,
    QStackedWidget, QTabWidget, QSplitter, QSlider, QListWidget, QListWidgetItem,
    QScrollArea
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

# --- DEFAULTS ---
DEFAULT_PROJECT_ROOT = str(Path(__file__).resolve().parent)
venv1 = Path("/home/joshua/venvs/ads_torch2")
venv2 = Path("/home/andreia/venvs/openads")

if venv1.exists():
    DEFAULT_VENV_PATH = str(venv1)
else:
    DEFAULT_VENV_PATH = str(venv2)
    
# Pipeline stage definitions from JSON
PIPELINE_STAGES = {
    "DWI": {
        "prepdata": {
            "label": "Prep Data",
            "inputs": ["Raw DICOM/NIfTI (DWI, ADC, B0)"],
            "outputs": [
                "preprocess/{id}_DWI.nii.gz",
                "preprocess/{id}_ADC.nii.gz",
                "preprocess/{id}_B0.nii.gz"
            ],
            "requirements": "Required: DWI\nOptional: ADC, B0, stroke mask"
        },
        "gen_mask": {
            "label": "Brain Mask",
            "inputs": [
                "preprocess/{id}_DWI.nii.gz"
            ],
            "outputs": [
                "preprocess/{id}_DWIbrain-mask.nii.gz"
            ],
            "requirements": "Auto-loaded: preprocessed DWI from stage 1\nCan replace by drag & drop"
        },
        "skull_strip": {
            "label": "Skull Strip",
            "inputs": [
                "preprocess/{id}_DWI.nii.gz",
                "preprocess/{id}_ADC.nii.gz",
                "preprocess/{id}_B0.nii.gz",
                "preprocess/{id}_DWIbrain-mask.nii.gz"
            ],
            "outputs": [
                "preprocess/{id}_DWI_brain.nii.gz",
                "preprocess/{id}_ADC_brain.nii.gz",
                "preprocess/{id}_B0_brain.nii.gz"
            ],
            "requirements": "Auto-loaded: preprocessed images + brain mask from stages 1-2\nCan replace by drag & drop"
        },
        "registration": {
            "label": "Registration",
            "inputs": [
                "preprocess/{id}_DWI_brain.nii.gz",
                "preprocess/{id}_ADC_brain.nii.gz",
                "preprocess/{id}_B0_brain.nii.gz"
            ],
            "outputs": [
                "registration/{id}_DWI_space-MNI152_aff.nii.gz",
                "registration/{id}_ADC_space-MNI152_aff.nii.gz",
                "registration/{id}_DWI_space-MNI152_aff_desc-norm.nii.gz",
                "registration/{id}_ADC_space-MNI152_aff_desc-norm.nii.gz",
                "registration/{id}_DWIbrain-mask_space-MNI152_aff.nii.gz",
                "registration/{id}_DWI_space-MNI152_affsyn.nii.gz",
                "registration/{id}_ADC_space-MNI152_affsyn.nii.gz",
                "registration/{id}_DWIbrain-mask_space-MNI152_affsyn.nii.gz",
                "registration/{id}_aff_space-individual2MNI152.mat",
                "registration/{id}_invaff_space-MNI1522individual.mat",
                "registration/{id}_syn_space-MNI1522MNI152.mat",
                "registration/{id}_warp_space-MNI1522MNI152.nii.gz",
                "registration/{id}_invsyn_space-MNI1522MNI152.mat",
                "registration/{id}_invwarp_space-MNI1522MNI152.nii.gz"
            ],
            "requirements": "Auto-loaded: Skull-stripped images from stage 3\nCan replace by drag & drop"
        },
        "inference": {
            "label": "Lesion Seg",
            "inputs": [
                "registration/{id}_DWI_space-MNI152_aff.nii.gz",
                "registration/{id}_ADC_space-MNI152_aff.nii.gz",
                "registration/{id}_DWIbrain-mask_space-MNI152_aff.nii.gz"
            ],
            "outputs": [
                "segment/{id}_stroke-mask_space-MNI152.nii.gz",
                "segment/{id}_stroke-mask_space-MNI152_affsyn.nii.gz",
                "segment/{id}_stroke-mask.nii.gz",
                "segment/{id}_metrics.json"
            ],
            "requirements": "Auto-loaded: Registered images from stage 4\nCan replace by drag & drop"
        },
        # "postprocessing": {
        #     "label": "Postprocess",
        #     "inputs": [
        #         "segmentation/{id}_stroke-mask_space-MNI152.nii.gz",
        #         "registration/{id}_invsyn_space-MNI1522MNI152.mat"
        #     ],
        #     "outputs": [
        #         "segmentation/{id}_stroke-mask.nii.gz",
        #         "segmentation/{id}_stroke-mask_space-MNI152_binary.nii.gz"
        #     ],
        #     "requirements": "Auto-loaded: Segmentation + transform from stages 4-5\nCan replace by drag & drop"
        # },
        "report": {
            "label": "Reporting",
            "inputs": [
                "segment/{id}_stroke-mask_space-MNI152_affsyn.nii.gz",
                "registration/{id}_ADC_space-MNI152_affsyn.nii.gz"
            ],
            "outputs": [
                "reporting/{id}_VASCULAR_QFV.csv",
                "reporting/{id}_LOBE_QFV.csv",
                "reporting/{id}_ASPECTS_QFV.csv",
                "reporting/{id}_ASPECTSPC_QFV.csv",
                "reporting/{id}_VENTRICLES_QFV.csv",
                "reporting/{id}_automatic_radiological_report.txt",
                "reporting/{id}_DWIstroke_space-MNI152.png",
                "reporting/{id}_DWIstroke.png",
                "reporting/{id}_VASCULAR_report_interpretation.pdf",
                "reporting/{id}_LOBE_report_interpretation.pdf",
                "reporting/{id}_ASPECTS_report_interpretation.pdf",
                "reporting/{id}_ASPECTSPC_report_interpretation.pdf"
            ],
            "requirements": "Auto-loaded: Segmentation + ADC from stages 4-5\nCan replace by drag & drop"
        }
    },
    "PWI": {
        "prepdata": {
            "label": "Prep Data",
            "inputs": ["Raw PWI ({id}_PWI.nii.gz)"],
            "outputs": ["preprocess/{id}_PWI.nii.gz"],
            "requirements": "Required: PWI series\nOptional: Additional sequences"
        },
        "gen_mask": {
            "label": "Brain Mask",
            "inputs": [
                "preprocess/{id}_DWI.nii.gz"
            ],
            "outputs": [
                "preprocess/{id}_DWIbrain-mask.nii.gz"
            ],
            "requirements": "Auto-loaded: preprocessed DWI from stage 1\nCan replace by drag & drop"
        },
        "skull_strip": {
            "label": "Skull Strip",
            "inputs": [
                "preprocess/{id}_DWI.nii.gz",
                "preprocess/{id}_ADC.nii.gz",
                "preprocess/{id}_DWIbrain-mask.nii.gz"
            ],
            "outputs": [
                "preprocess/{id}_DWI_brain.nii.gz",
                "preprocess/{id}_ADC_brain.nii.gz"
            ],
            "requirements": "Auto-loaded: preprocessed images + brain mask from stages 1-2\nCan replace by drag & drop"
        },
        "gen_ttp": {
            "label": "Generate TTP",
            "inputs": ["preprocess/{id}_PWI.nii.gz"],
            "outputs": ["preprocess/{id}_TTP.nii.gz"],
            "requirements": "Auto-loaded: PWI from stage 1\nCan replace by drag & drop"
        },
        "registration": {
            "label": "Registration",
            "inputs": [
                "preprocess/{id}_DWI_brain.nii.gz",
                "preprocess/{id}_ADC_brain.nii.gz",
                "preprocess/{id}_DWIbrain-mask.nii.gz"
            ],
            "outputs": [
                "registration/{id}_DWI_space-MNI152_aff.nii.gz",
                "registration/{id}_ADC_space-MNI152_aff.nii.gz",
                "registration/{id}_DWIbrain-mask_space-MNI152_aff.nii.gz",
                "registration/{id}_DWI_space-MNI152_aff_desc-norm.nii.gz",
                "registration/{id}_ADC_space-MNI152_aff_desc-norm.nii.gz",
                "registration/{id}_aff_space-individual2MNI152.mat",
                "registration/{id}_invaff_space-MNI1522individual.mat",
                "registration/{id}_syn_space-MNI1522MNI152.mat",
                "registration/{id}_warp_space-MNI1522MNI152.nii.gz",
                "registration/{id}_DWI_space-MNI152_affsyn.nii.gz",
                "registration/{id}_ADC_space-MNI152_affsyn.nii.gz",
                "registration/{id}_DWIbrain-mask_space-MNI152_affsyn.nii.gz"
            ],
            "requirements": "Auto-loaded: DWI/ADC brain + mask from stages 2-3\nCan replace by drag & drop"
        },
        "ttpadc_coreg": {
            "label": "TTP-ADC Coreg",
            "inputs": [
                "preprocess/{id}_TTP.nii.gz",
                "preprocess/{id}_PWI.nii.gz",
                "preprocess/{id}_PWIbrain-mask.nii.gz",
                "preprocess/{id}_DWIbrain-mask.nii.gz",
                "registration/{id}_aff_space-individual2MNI152.mat"
            ],
            "outputs": [
                "registration/{id}_aff_space-individualTTP2ADC.mat",
                "registration/{id}_invaff_space-ADC2individualTTP.mat",
                "registration/{id}_TTP_space-DWI.nii.gz",
                "registration/{id}_TTP_space-MNI152_aff.nii.gz",
                "registration/{id}_TTP_space-MNI152_aff_desc-norm.nii.gz",
                "registration/{id}_TTP_space-MNI152_affsyn.nii.gz",
                "registration/{id}_HP_manual_space-MNI152_aff.nii.gz",
                "registration/{id}_HP_manual_space-MNI152_affsyn.nii.gz"
            ],
            "requirements": "Auto-loaded: TTP + transform from stages 4-5\nCan replace by drag & drop"
        },
        "inference": {
            "label": "Inference",
            "inputs": [
                "registration/{id}_DWI_space-MNI152_aff.nii.gz",
                "registration/{id}_ADC_space-MNI152_aff.nii.gz",
                "registration/{id}_TTP_space-MNI152_aff.nii.gz"
            ],
            "outputs": [
                "segmentation/{id}_HPsymclassic-mask_space-MNI152.nii.gz",
                "segmentation/{id}_HP-mask_space-MNI152.nii.gz",
                "segmentation/{id}_HP-mask_space-MNI152_affsyn.nii.gz",
                "segmentation/{id}_HP-mask.nii.gz",
                "segmentation/{id}_metrics.json"
            ],
            "requirements": "Auto-loaded: Registered images from stages 5-6\nCan replace by drag & drop"
        },
        "report": {
            "label": "Reporting",
            "inputs": [
                "registration/{id}_TTP_space-MNI152_affsyn.nii.gz",
                "segmentation/{id}_HP-mask_space-MNI152_affsyn.nii.gz"
            ],
            "outputs": [
                "reporting/{id}_automatic_radiological_report.txt",
                "reporting/{id}_Vascular_HPload.csv",
                "reporting/{id}_Vascular_HPQFV.csv",
                "reporting/{id}_Lobe_HPload.csv",
                "reporting/{id}_Lobe_HPQFV.csv",
                "reporting/{id}_Aspects_HPload.csv",
                "reporting/{id}_Aspects_HPQFV.csv",
                "reporting/{id}_AspectsPC_HPload.csv",
                "reporting/{id}_AspectsPC_HPQFV.csv",
                "reporting/{id}_HP_space-MNI152.png",
                "reporting/{id}_HP.png"
            ],
            "requirements": "Auto-loaded: Registered images from stages 5-6\nCan replace by drag & drop"
        }
    }
}

# Better color scheme
COLORS = {
    "primary": "#1976D2",
    "success": "#388E3C",
    "danger": "#D32F2F",
    "warning": "#F57C00",
    "info": "#0288D1",
    "secondary": "#757575",
}


# -------------------- HELPER: NIFTI VIEWER --------------------
class NiftiViewer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scan_dir: Optional[Path] = None
        self.subject_id: str = ""
        self._base_vol = None
        self._ovl_vol = None
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        
        r1 = QHBoxLayout()
        r1.addWidget(QLabel("Stage View:"))
        self.stage_combo = QComboBox()
        self.stage_combo.addItems(["Registration Check", "Segmentation Check", "Raw Data"])
        r1.addWidget(self.stage_combo, 1)
        b = QPushButton("Refresh Folder")
        b.setStyleSheet(f"background-color: {COLORS['info']}; color: white; padding: 5px; border-radius: 4px;")
        b.clicked.connect(self._refresh_clicked)
        r1.addWidget(b)
        root.addLayout(r1)

        form = QFormLayout()
        self.base_combo = QComboBox()
        self.ovl_combo = QComboBox()
        self.ovl_combo.addItem("None")
        form.addRow("Base Image:", self.base_combo)
        form.addRow("Overlay:", self.ovl_combo)
        root.addLayout(form)

        btn_load = QPushButton("Load Selected Images")
        btn_load.setStyleSheet(f"background-color: {COLORS['primary']}; color: white; padding: 8px; border-radius: 4px; font-weight: bold;")
        btn_load.clicked.connect(self._load_clicked)
        root.addWidget(btn_load)

        ctrl = QHBoxLayout()
        self.orient_combo = QComboBox()
        self.orient_combo.addItems(["Axial (Z)", "Coronal (Y)", "Sagittal (X)"])
        self.orient_combo.currentIndexChanged.connect(self._update_view)
        
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.valueChanged.connect(self._update_view)
        
        ctrl.addWidget(QLabel("View:"))
        ctrl.addWidget(self.orient_combo)
        ctrl.addWidget(self.slice_slider, 1)
        root.addLayout(ctrl)

        self.fig, self.ax = plt.subplots(figsize=(6, 6), dpi=100)
        self.fig.patch.set_facecolor("#f0f0f0")
        self.canvas = FigureCanvas(self.fig)
        root.addWidget(self.canvas, 1)

    def update_context(self, scan_dir: Path, subject_id: str):
        self.scan_dir = scan_dir
        self.subject_id = subject_id
        self._refresh_clicked()

    def _refresh_clicked(self):
        if not self.scan_dir or not self.scan_dir.exists():
            return
        files = sorted([str(p.relative_to(self.scan_dir)) for p in self.scan_dir.rglob("*.nii.gz")])
        curr_base = self.base_combo.currentText()
        curr_ovl = self.ovl_combo.currentText()
        self.base_combo.clear()
        self.ovl_combo.clear()
        self.base_combo.addItems(files)
        self.ovl_combo.addItem("None")
        self.ovl_combo.addItems(files)
        if curr_base in files:
            self.base_combo.setCurrentText(curr_base)
        if curr_ovl in files:
            self.ovl_combo.setCurrentText(curr_ovl)

    def _load_clicked(self):
        if not self.scan_dir:
            return
        base_rel = self.base_combo.currentText()
        if not base_rel:
            return
        try:
            p = self.scan_dir / base_rel
            img = nib.as_closest_canonical(nib.load(str(p)))
            data = img.get_fdata(dtype=np.float32)
            valid = data[np.isfinite(data)]
            if valid.size > 0:
                lo, hi = np.percentile(valid, [1, 99])
                data = np.clip((data - lo) / (hi - lo + 1e-8), 0, 1)
            self._base_vol = data
            ovl_rel = self.ovl_combo.currentText()
            if ovl_rel != "None":
                p_ovl = self.scan_dir / ovl_rel
                img_ovl = nib.as_closest_canonical(nib.load(str(p_ovl)))
                self._ovl_vol = (img_ovl.get_fdata() > 0).astype(np.float32)
            else:
                self._ovl_vol = None
            dim = {0: 2, 1: 1, 2: 0}[self.orient_combo.currentIndex()]
            self.slice_slider.setRange(0, self._base_vol.shape[dim] - 1)
            self.slice_slider.setValue(self._base_vol.shape[dim] // 2)
            self._update_view()
        except Exception as e:
            print(f"Error loading image: {e}")

    def _update_view(self):
        if self._base_vol is None:
            return
        idx = self.slice_slider.value()
        mode = self.orient_combo.currentIndex()
        axis_map = {0: 2, 1: 1, 2: 0}
        axis = axis_map[mode]
        if idx >= self._base_vol.shape[axis]:
            idx = self._base_vol.shape[axis] - 1
        if axis == 2:
            slice_img = self._base_vol[:, :, idx]
        elif axis == 1:
            slice_img = self._base_vol[:, idx, :]
        else:
            slice_img = self._base_vol[idx, :, :]
        slice_img = np.rot90(slice_img)
        self.ax.clear()
        self.ax.imshow(slice_img, cmap="gray")
        if self._ovl_vol is not None and self._ovl_vol.shape == self._base_vol.shape:
            if axis == 2:
                ovl_sl = self._ovl_vol[:, :, idx]
            elif axis == 1:
                ovl_sl = self._ovl_vol[:, idx, :]
            else:
                ovl_sl = self._ovl_vol[idx, :, :]
            ovl_sl = np.rot90(ovl_sl)
            self.ax.imshow(np.ma.masked_where(ovl_sl < 0.5, ovl_sl), cmap="autumn", alpha=0.5)
        self.ax.axis('off')
        self.canvas.draw()


# -------------------- DRAG DROP AREA --------------------
class DragDropArea(QLabel):
    filesDropped = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setText("Drag & Drop Files Here\n(or anywhere in this window)")
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(
            "border: 3px dashed #888; padding: 40px; font-size: 14pt; "
            "background: #fdfdfd; color: #555; border-radius: 8px;"
        )
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        paths = [url.toLocalFile() for url in event.mimeData().urls() if url.toLocalFile()]
        if paths:
            self.filesDropped.emit(paths)


# -------------------- SIMPLE NIFTI VIEWER --------------------
class SimpleNiftiSliceViewer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.fig, self.ax = plt.subplots(figsize=(4, 4), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

    def show_nifti(self, path: str):
        self.ax.clear()
        try:
            img = nib.as_closest_canonical(nib.load(path))
            data = img.get_fdata(dtype=np.float32)
            valid = data[np.isfinite(data)]
            if valid.size > 0:
                lo, hi = np.percentile(valid, [1, 99])
                data = np.clip((data - lo) / (hi - lo + 1e-8), 0, 1)
            z = data.shape[2] // 2 if data.ndim == 3 else 0
            slice_img = np.rot90(data[:, :, z])
            self.ax.imshow(slice_img, cmap="gray")
            self.ax.axis("off")
        except Exception as e:
            self.ax.text(0.5, 0.5, f"Error:\n{e}", ha="center", va="center", fontsize=9)
            self.ax.axis("off")
        self.canvas.draw()


# -------------------- STAGE UPLOAD WIDGET --------------------
class StageUploadWidget(QWidget):
    def __init__(
        self,
        modality: str,
        stage_key: str,
        stage_info: Dict,
        log_callback: Callable[[str], None],
        get_subject_path: Callable[[], str],
        get_config_path: Callable[[], str],
        get_gpu: Callable[[], str],
        get_output_root: Callable[[], str],
        parent=None
    ):
        super().__init__(parent)
        self.modality = modality
        self.stage_key = stage_key
        self.stage_info = stage_info
        self.stage_label = stage_info["label"]
        self.log_callback = log_callback
        self.get_subject_path = get_subject_path
        self.get_config_path = get_config_path
        self.get_gpu = get_gpu
        self.get_output_root = get_output_root
        self.uploaded_files: List[str] = []
        self.output_files: List[str] = []
        self.setAcceptDrops(True)
        self._build()
        
        QTimer.singleShot(500, self._auto_load_files)

    def _build(self):
        layout = QVBoxLayout(self)

        title = QLabel(f"<b>{self.stage_label}</b> ({self.modality})")
        title.setStyleSheet(f"font-size: 14pt; color: {COLORS['primary']};")
        layout.addWidget(title)

        self.req_label = QLabel(self.stage_info.get("requirements", ""))
        self.req_label.setWordWrap(True)
        self.req_label.setStyleSheet("font-size: 11pt; padding: 8px; background: #e3f2fd; border-radius: 4px;")
        layout.addWidget(self.req_label)

        self.drag_area = DragDropArea()
        self.drag_area.filesDropped.connect(self._on_files_dropped)
        layout.addWidget(self.drag_area)

        hl = QHBoxLayout()
        
        left = QVBoxLayout()
        
        upload_header = QHBoxLayout()
        upload_header.addWidget(QLabel("<b>Input Files:</b>"))
        btn_auto_load = QPushButton("Auto-Load")
        btn_auto_load.setStyleSheet(f"background-color: {COLORS['info']}; color: white; padding: 4px 8px; border-radius: 4px;")
        btn_auto_load.clicked.connect(self._auto_load_files)
        upload_header.addWidget(btn_auto_load)
        upload_header.addStretch()
        left.addLayout(upload_header)
        
        self.upload_list = QListWidget()
        self.upload_list.itemDoubleClicked.connect(self._on_upload_double_click)
        left.addWidget(self.upload_list, 1)
        
        btn_remove = QPushButton("Remove Selected")
        btn_remove.setStyleSheet(f"background-color: {COLORS['danger']}; color: white; padding: 6px; border-radius: 4px;")
        btn_remove.clicked.connect(self._remove_uploaded)
        left.addWidget(btn_remove)
        
        left.addWidget(QLabel("<b>Output Files:</b>"))
        self.output_list = QListWidget()
        self.output_list.itemDoubleClicked.connect(self._on_output_double_click)
        left.addWidget(self.output_list, 1)
        
        btn_refresh = QPushButton("Refresh Output")
        btn_refresh.setStyleSheet(f"background-color: {COLORS['info']}; color: white; padding: 6px; border-radius: 4px;")
        btn_refresh.clicked.connect(self._refresh_output_files)
        left.addWidget(btn_refresh)
        
        hl.addLayout(left, 1)

        right = QVBoxLayout()
        right.addWidget(QLabel("<b>Preview:</b>"))
        self.preview_stack = QStackedWidget()

        self.nifti_viewer = SimpleNiftiSliceViewer()
        self.preview_stack.addWidget(self.nifti_viewer)

        self.image_label = QLabel("No image")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setScaledContents(True)
        img_container = QWidget()
        img_lay = QVBoxLayout(img_container)
        img_lay.addWidget(self.image_label)
        self.preview_stack.addWidget(img_container)

        self.text_view = QTextEdit()
        self.text_view.setReadOnly(True)
        self.preview_stack.addWidget(self.text_view)

        self.info_label = QLabel("Double-click a file to preview")
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet("font-size: 11pt; color: #666;")
        self.preview_stack.addWidget(self.info_label)

        self.preview_stack.setCurrentWidget(self.info_label)
        right.addWidget(self.preview_stack, 1)
        hl.addLayout(right, 2)

        layout.addLayout(hl)

        self.run_btn = QPushButton("Run This Stage")
        self.run_btn.setMinimumHeight(50)
        self.run_btn.setStyleSheet(
            f"background-color: {COLORS['success']}; color: white; font-weight: bold; "
            "font-size: 13pt; border-radius: 6px;"
        )
        self.run_btn.clicked.connect(self._on_run_clicked)
        layout.addWidget(self.run_btn)

    def _auto_load_files(self):
        """Auto-load expected input files based on stage definition"""
        subject_path = self.get_subject_path()
        if not subject_path:
            return
        
        output_root = self.get_output_root()
        subject_name = Path(subject_path).name
        
        self.upload_list.clear()
        self.uploaded_files.clear()
        
        # Load files based on stage inputs
        for input_pattern in self.stage_info.get("inputs", []):
            if "Raw" in input_pattern or "DICOM" in input_pattern:
                continue
            
            # Replace {id} with subject name
            input_path = input_pattern.replace("{id}", subject_name)
            full_path = Path(output_root) / subject_name / self.modality / input_path
            
            # Handle wildcards
            if "*" in input_path:
                parent_dir = full_path.parent
                pattern = full_path.name
                if parent_dir.exists():
                    for fpath in parent_dir.glob(pattern):
                        if fpath.is_file():
                            self._add_file_to_list(str(fpath))
            else:
                if full_path.exists():
                    self._add_file_to_list(str(full_path))
        
        if self.uploaded_files:
            self.log_callback(
                f"<span style='color: {COLORS['info']};'>[AUTO-LOAD]</span> {self.stage_label}: "
                f"Loaded {len(self.uploaded_files)} file(s)"
            )

    def _add_file_to_list(self, fpath: str):
        """Add file to uploaded list"""
        if fpath not in self.uploaded_files:
            self.uploaded_files.append(fpath)
            item = QListWidgetItem(Path(fpath).name)
            item.setToolTip(fpath)
            item.setData(Qt.UserRole, fpath)
            self.upload_list.addItem(item)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        paths = [url.toLocalFile() for url in event.mimeData().urls() 
                if url.toLocalFile() and os.path.isfile(url.toLocalFile())]
        if paths:
            self._on_files_dropped(paths)

    def _on_files_dropped(self, paths: List[str]):
        for p in paths:
            self._add_file_to_list(p)
        self.log_callback(
            f"<span style='color: {COLORS['success']};'>[UPLOAD]</span> {self.stage_label}: "
            f"Added {len(paths)} file(s)"
        )

    def _remove_uploaded(self):
        current = self.upload_list.currentItem()
        if current:
            path = current.data(Qt.UserRole)
            self.uploaded_files.remove(path)
            self.upload_list.takeItem(self.upload_list.row(current))
            self.log_callback(f"<span style='color: {COLORS['danger']};'>[REMOVE]</span> {os.path.basename(path)}")

    def _refresh_output_files(self):
        """Scan output directory and list generated files"""
        subject_path = self.get_subject_path()
        if not subject_path:
            self.log_callback(f"<span style='color: {COLORS['warning']};'>[WARNING]</span> No subject selected")
            return
        
        output_root = self.get_output_root()
        subject_name = Path(subject_path).name
        
        self.output_list.clear()
        self.output_files.clear()
        
        # Check expected output files
        expected_outputs = []
        for output_pattern in self.stage_info.get("outputs", []):
            output_path = output_pattern.replace("{id}", subject_name)
            full_path = Path(output_root) / subject_name / self.modality / output_path
            
            if "*" in output_path:
                parent_dir = full_path.parent
                pattern = full_path.name
                if parent_dir.exists():
                    for fpath in parent_dir.glob(pattern):
                        if fpath.is_file():
                            expected_outputs.append(fpath)
            else:
                if full_path.exists():
                    expected_outputs.append(full_path)
        
        # Add to output list
        for fpath in expected_outputs:
            self.output_files.append(str(fpath))
            item = QListWidgetItem(fpath.name)
            item.setToolTip(str(fpath))
            item.setData(Qt.UserRole, str(fpath))
            self.output_list.addItem(item)
        
        if expected_outputs:
            self.log_callback(
                f"<span style='color: {COLORS['success']};'>[OUTPUT]</span> "
                f"{self.stage_label}: Found {len(expected_outputs)} output file(s)"
            )
        else:
            self.log_callback(
                f"<span style='color: {COLORS['warning']};'>[OUTPUT]</span> "
                f"{self.stage_label}: No output files found yet"
            )

    def _on_upload_double_click(self, item):
        self._preview_file(item.data(Qt.UserRole))

    def _on_output_double_click(self, item):
        self._preview_file(item.data(Qt.UserRole))

    def _preview_file(self, path: str):
        if not os.path.exists(path):
            self.preview_stack.setCurrentWidget(self.info_label)
            self.info_label.setText("File not found")
            return

        ext = '.nii.gz' if path.endswith('.nii.gz') else os.path.splitext(path)[1].lower()
        
        if ext in (".nii", ".nii.gz"):
            self.preview_stack.setCurrentWidget(self.nifti_viewer)
            self.nifti_viewer.show_nifti(path)
        elif ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
            pix = QPixmap(path)
            if not pix.isNull():
                self.image_label.setPixmap(pix)
                self.preview_stack.setCurrentWidget(self.image_label.parentWidget())
            else:
                self.info_label.setText("Cannot load image")
                self.preview_stack.setCurrentWidget(self.info_label)
        elif ext == ".csv":
            try:
                with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
                    rows = list(csv.reader(f))
                if not rows:
                    self.text_view.setHtml("<i>Empty CSV file</i>")
                else:
                    header = rows[0]
                    body = rows[1:51]
                    html = [
                        "<h3 style='margin:0 0 8px 0;'>CSV Preview</h3>",
                        "<table border='1' cellspacing='0' cellpadding='4' style='border-collapse:collapse;font-size:12px;'>",
                        "<tr>" + "".join(f"<th>{escape(c)}</th>" for c in header) + "</tr>",
                    ]
                    for row in body:
                        html.append("<tr>" + "".join(f"<td>{escape(c)}</td>" for c in row) + "</tr>")
                    html.append("</table>")
                    if len(rows) > 51:
                        html.append(f"<p><i>Showing first {len(body)} rows only.</i></p>")
                    self.text_view.setHtml("".join(html))
                self.preview_stack.setCurrentWidget(self.text_view)
            except Exception as e:
                self.info_label.setText(f"Error reading file:\n{e}")
                self.preview_stack.setCurrentWidget(self.info_label)
        elif ext in (".txt", ".log", ".cfg", ".ini", ".yaml", ".yml"):
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    txt = f.read(30000)
                self.text_view.setHtml(
                    "<h3 style='margin:0 0 8px 0;'>Text Preview</h3>"
                    f"<pre style='font-family:monospace; font-size:12px; white-space:pre-wrap;'>{escape(txt)}</pre>"
                )
                self.preview_stack.setCurrentWidget(self.text_view)
            except Exception as e:
                self.info_label.setText(f"Error reading file:\n{e}")
                self.preview_stack.setCurrentWidget(self.info_label)
        elif ext == ".pdf":
            p = Path(path)
            self.text_view.setHtml(
                "<h3 style='margin:0 0 8px 0;'>PDF Report</h3>"
                f"<p><b>File:</b> {escape(p.name)}</p>"
                f"<p><b>Path:</b> {escape(str(p))}</p>"
                f"<p><b>Size:</b> {p.stat().st_size / 1024.0:.1f} KB</p>"
                "<p><i>Embedded PDF preview is not enabled in this GUI. Open the file from the reporting folder.</i></p>"
            )
            self.preview_stack.setCurrentWidget(self.text_view)
        else:
            self.info_label.setText(f"Preview not supported:\n{os.path.basename(path)}")
            self.preview_stack.setCurrentWidget(self.info_label)

    def _on_run_clicked(self):
        subject_path = self.get_subject_path()
        if not subject_path:
            QMessageBox.warning(self, "No Subject", "Please select a subject folder first.")
            return

        config_path = self.get_config_path()
        gpu = self.get_gpu()
        
        script_name = f"run_ads_{self.modality.lower()}.py"
        script_path = Path(DEFAULT_PROJECT_ROOT) / "scripts" / script_name
        
        if not script_path.exists():
            QMessageBox.warning(self, "Script Not Found", f"Pipeline script not found:\n{script_path}")
            return

        # Build command with auto-close and output refresh
        output_root = self.get_output_root().strip()
        cmd = (
            f"cd \"{DEFAULT_PROJECT_ROOT}\" && "
            f"source \"{DEFAULT_VENV_PATH}/bin/activate\" && "
            f"python \"{script_path}\" "
            f"--subject-path \"{subject_path}\" "
            f"--config \"{config_path}\" "
            f"--gpu {gpu} "
            f"--stages {self.stage_key} "
            f"{f'--output-root \"{output_root}\" ' if output_root else ''}"
            f"&& "
            f"echo '' && echo 'Stage completed successfully!' && sleep 2 && exit"
        )

        # Very small terminal (1/4 size)
        term_cmd = []
        if shutil.which("gnome-terminal"):
            term_cmd = ["gnome-terminal", "--geometry=60x15", "--", "bash", "-c", cmd]
        elif shutil.which("konsole"):
            term_cmd = ["konsole", "--geometry", "600x300", "-e", "bash", "-c", cmd]
        elif shutil.which("xterm"):
            term_cmd = ["xterm", "-geometry", "60x15", "-e", "bash", "-c", cmd]
        else:
            QMessageBox.critical(self, "Error", "No compatible terminal found")
            return

        try:
            subprocess.Popen(term_cmd)
            self.log_callback(
                f"<span style='color: {COLORS['success']}; font-weight: bold;'>[LAUNCHED]</span> "
                f"{self.stage_label}"
            )
            # Auto-refresh output after 10 seconds
            QTimer.singleShot(10000, self._refresh_output_files)
        except Exception as e:
            QMessageBox.critical(self, "Launch Failed", str(e))


# -------------------- MODALITY PAGE --------------------
class ModalityPage(QWidget):
    def __init__(self, modality: str):
        super().__init__()
        self.modality = modality
        self.stage_checks: Dict[str, QCheckBox] = {}
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        
        box = QGroupBox("Input / Output")
        form = QFormLayout(box)

        self.subject_dir_edit = QLineEdit()
        b1 = QPushButton("Browse")
        b1.setStyleSheet(f"background-color: {COLORS['secondary']}; color: white; padding: 5px; border-radius: 4px;")
        b1.clicked.connect(lambda: self._browse_dir(self.subject_dir_edit))
        
        r1 = QHBoxLayout()
        r1.addWidget(self.subject_dir_edit)
        r1.addWidget(b1)
        form.addRow("Raw Subject Folder:", r1)

        self.subject_id_edit = QLineEdit()
        self.subject_dir_edit.textChanged.connect(self._auto_fill_id)
        form.addRow("Subject ID:", self.subject_id_edit)

        self.output_root_edit = QLineEdit(f"{DEFAULT_PROJECT_ROOT}/output")
        b2 = QPushButton("Browse")
        b2.setStyleSheet(f"background-color: {COLORS['secondary']}; color: white; padding: 5px; border-radius: 4px;")
        b2.clicked.connect(lambda: self._browse_dir(self.output_root_edit))
        r2 = QHBoxLayout()
        r2.addWidget(self.output_root_edit)
        r2.addWidget(b2)
        form.addRow("Output Root:", r2)
        
        layout.addWidget(box)

        box2 = QGroupBox(f"Pipeline Stages ({self.modality})")
        grid = QGridLayout(box2)
        
        for i, (key, info) in enumerate(PIPELINE_STAGES[self.modality].items()):
            c = QCheckBox(info["label"])
            c.setChecked(True)
            grid.addWidget(c, i // 2, i % 2)
            self.stage_checks[key] = c
        layout.addWidget(box2)

        adv = QGroupBox("Configuration")
        f2 = QFormLayout(adv)
        
        default_cfg = f"configs/{self.modality.lower()}_pipeline.yaml"
        self.config_edit = QLineEdit(default_cfg)
        f2.addRow("Config File:", self.config_edit)
        
        self.gpu_edit = QLineEdit("1")
        f2.addRow("GPU ID:", self.gpu_edit)
        
        layout.addWidget(adv)
        layout.addStretch(1)

    def _browse_dir(self, line_edit):
        d = QFileDialog.getExistingDirectory(self, "Select Directory", line_edit.text() or DEFAULT_PROJECT_ROOT)
        if d:
            line_edit.setText(d)

    def _auto_fill_id(self, text):
        if text:
            self.subject_id_edit.setText(Path(text).name)


class CombinedPage(QWidget):
    def __init__(self):
        super().__init__()
        self.dwi_stage_checks: Dict[str, QCheckBox] = {}
        self.pwi_stage_checks: Dict[str, QCheckBox] = {}
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)

        box = QGroupBox("Input / Output (Combined)")
        form = QFormLayout(box)

        self.dwi_subject_dir_edit = QLineEdit()
        b1 = QPushButton("Browse")
        b1.setStyleSheet(f"background-color: {COLORS['secondary']}; color: white; padding: 5px; border-radius: 4px;")
        b1.clicked.connect(lambda: self._browse_dir(self.dwi_subject_dir_edit))
        r1 = QHBoxLayout()
        r1.addWidget(self.dwi_subject_dir_edit)
        r1.addWidget(b1)
        form.addRow("DWI Subject Folder:", r1)

        self.pwi_subject_dir_edit = QLineEdit()
        b2 = QPushButton("Browse")
        b2.setStyleSheet(f"background-color: {COLORS['secondary']}; color: white; padding: 5px; border-radius: 4px;")
        b2.clicked.connect(lambda: self._browse_dir(self.pwi_subject_dir_edit))
        r2 = QHBoxLayout()
        r2.addWidget(self.pwi_subject_dir_edit)
        r2.addWidget(b2)
        form.addRow("PWI Subject Folder:", r2)

        self.subject_id_edit = QLineEdit()
        self.dwi_subject_dir_edit.textChanged.connect(self._sync_subject_id)
        self.pwi_subject_dir_edit.textChanged.connect(self._sync_subject_id)
        form.addRow("Subject ID:", self.subject_id_edit)

        self.output_root_edit = QLineEdit(f"{DEFAULT_PROJECT_ROOT}/output")
        b3 = QPushButton("Browse")
        b3.setStyleSheet(f"background-color: {COLORS['secondary']}; color: white; padding: 5px; border-radius: 4px;")
        b3.clicked.connect(lambda: self._browse_dir(self.output_root_edit))
        r3 = QHBoxLayout()
        r3.addWidget(self.output_root_edit)
        r3.addWidget(b3)
        form.addRow("Output Root:", r3)

        layout.addWidget(box)

        dwi_box = QGroupBox("Pipeline Stages (DWI)")
        dwi_grid = QGridLayout(dwi_box)
        dwi_keys = list(PIPELINE_STAGES["DWI"].keys())
        for i, key in enumerate(dwi_keys):
            c = QCheckBox(PIPELINE_STAGES["DWI"][key]["label"])
            c.setChecked(True)
            dwi_grid.addWidget(c, i // 2, i % 2)
            self.dwi_stage_checks[key] = c
        layout.addWidget(dwi_box)

        pwi_box = QGroupBox("Pipeline Stages (PWI)")
        pwi_grid = QGridLayout(pwi_box)
        pwi_keys = list(PIPELINE_STAGES["PWI"].keys())
        for i, key in enumerate(pwi_keys):
            c = QCheckBox(PIPELINE_STAGES["PWI"][key]["label"])
            c.setChecked(True)
            pwi_grid.addWidget(c, i // 2, i % 2)
            self.pwi_stage_checks[key] = c
        layout.addWidget(pwi_box)

        adv = QGroupBox("Configuration")
        f2 = QFormLayout(adv)
        self.dwi_config_edit = QLineEdit("configs/dwi_pipeline.yaml")
        f2.addRow("DWI Config File:", self.dwi_config_edit)
        self.pwi_config_edit = QLineEdit("configs/pwi_pipeline.yaml")
        f2.addRow("PWI Config File:", self.pwi_config_edit)
        self.gpu_edit = QLineEdit("1")
        f2.addRow("GPU ID:", self.gpu_edit)
        layout.addWidget(adv)
        layout.addStretch(1)

    def _browse_dir(self, line_edit):
        d = QFileDialog.getExistingDirectory(self, "Select Directory", line_edit.text() or DEFAULT_PROJECT_ROOT)
        if d:
            line_edit.setText(d)

    def _sync_subject_id(self, _text):
        dwi_id = Path(self.dwi_subject_dir_edit.text()).name if self.dwi_subject_dir_edit.text() else ""
        pwi_id = Path(self.pwi_subject_dir_edit.text()).name if self.pwi_subject_dir_edit.text() else ""
        self.subject_id_edit.setText(dwi_id or pwi_id)

    def selected_dwi_stages(self) -> List[str]:
        return [k for k, cb in self.dwi_stage_checks.items() if cb.isChecked()]

    def selected_pwi_stages(self) -> List[str]:
        return [k for k, cb in self.pwi_stage_checks.items() if cb.isChecked()]


class ReportFilesWidget(QWidget):
    """Dedicated reporting viewer (no drag/drop, no stage execution)."""

    def __init__(self, get_subject_id: Callable[[], str], get_output_root: Callable[[], str], parent=None):
        super().__init__(parent)
        self.get_subject_id = get_subject_id
        self.get_output_root = get_output_root
        self.output_files: List[str] = []
        self._build()
        QTimer.singleShot(500, self.refresh_files)

    def _build(self):
        layout = QVBoxLayout(self)
        title = QLabel("<b>Report Files (DWI + PWI)</b>")
        title.setStyleSheet(f"font-size: 14pt; color: {COLORS['primary']};")
        layout.addWidget(title)

        desc = QLabel("Automatically loads CSV / PDF / TXT / PNG from reporting folders.")
        desc.setStyleSheet("font-size: 11pt; padding: 6px; background: #e3f2fd; border-radius: 4px;")
        layout.addWidget(desc)

        top = QHBoxLayout()
        self.status_label = QLabel("No subject selected")
        top.addWidget(self.status_label, 1)
        btn_refresh = QPushButton("Refresh")
        btn_refresh.setStyleSheet(f"background-color: {COLORS['info']}; color: white; padding: 6px; border-radius: 4px;")
        btn_refresh.clicked.connect(self.refresh_files)
        top.addWidget(btn_refresh)
        layout.addLayout(top)

        split = QHBoxLayout()
        left = QVBoxLayout()
        left.addWidget(QLabel("<b>Reporting Files:</b>"))
        self.file_list = QListWidget()
        self.file_list.itemDoubleClicked.connect(self._on_file_double_click)
        left.addWidget(self.file_list, 1)
        split.addLayout(left, 1)

        right = QVBoxLayout()
        right.addWidget(QLabel("<b>Preview:</b>"))
        self.preview_stack = QStackedWidget()
        self.text_view = QTextEdit()
        self.text_view.setReadOnly(True)
        self.preview_stack.addWidget(self.text_view)

        self.image_label = QLabel("No image")
        self.image_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.image_label.setScaledContents(False)
        self.image_scroll = QScrollArea()
        self.image_scroll.setWidgetResizable(False)
        self.image_scroll.setWidget(self.image_label)
        self.preview_stack.addWidget(self.image_scroll)

        self.info_label = QLabel("Double-click a file to preview")
        self.info_label.setAlignment(Qt.AlignCenter)
        self.preview_stack.addWidget(self.info_label)
        self.preview_stack.setCurrentWidget(self.info_label)
        right.addWidget(self.preview_stack, 1)
        split.addLayout(right, 2)
        layout.addLayout(split)

    def refresh_files(self):
        sid = self.get_subject_id().strip()
        out_root = self.get_output_root().strip()
        self.file_list.clear()
        self.output_files.clear()
        if not sid or not out_root:
            self.status_label.setText("No subject selected")
            return

        # Always resolve reporting files from the user-selected Output Root.
        root = Path(out_root).expanduser().resolve() / sid
        report_dirs = [root / "DWI" / "reporting", root / "PWI" / "reporting"]
        files = []
        for rd in report_dirs:
            if rd.exists():
                for ext in ("*.txt", "*.csv", "*.pdf", "*.png"):
                    files.extend(sorted(rd.glob(ext)))
        for p in files:
            self.output_files.append(str(p))
            rel = p.relative_to(root) if root in p.parents else p
            item = QListWidgetItem(str(rel))
            item.setToolTip(str(p))
            item.setData(Qt.UserRole, str(p))
            self.file_list.addItem(item)
        self.status_label.setText(f"{sid}: {len(files)} reporting file(s)")

    def _on_file_double_click(self, item):
        self._preview_file(item.data(Qt.UserRole))

    def _preview_file(self, path: str):
        if not os.path.exists(path):
            self.info_label.setText("File not found")
            self.preview_stack.setCurrentWidget(self.info_label)
            return
        ext = '.nii.gz' if path.endswith('.nii.gz') else os.path.splitext(path)[1].lower()
        if ext == ".csv":
            try:
                with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
                    rows = list(csv.reader(f))
                if not rows:
                    self.text_view.setHtml("<i>Empty CSV file</i>")
                else:
                    header = rows[0]
                    body = rows[1:51]
                    html = [
                        "<h3 style='margin:0 0 8px 0;'>CSV Preview</h3>",
                        "<table border='1' cellspacing='0' cellpadding='4' style='border-collapse:collapse;font-size:12px;'>",
                        "<tr>" + "".join(f"<th>{escape(c)}</th>" for c in header) + "</tr>",
                    ]
                    for row in body:
                        html.append("<tr>" + "".join(f"<td>{escape(c)}</td>" for c in row) + "</tr>")
                    html.append("</table>")
                    self.text_view.setHtml("".join(html))
                self.preview_stack.setCurrentWidget(self.text_view)
            except Exception as e:
                self.info_label.setText(f"Error reading CSV:\n{e}")
                self.preview_stack.setCurrentWidget(self.info_label)
        elif ext in (".txt", ".log"):
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    txt = f.read(30000)
                self.text_view.setHtml(
                    "<h3 style='margin:0 0 8px 0;'>Text Preview</h3>"
                    f"<pre style='font-family:monospace; font-size:12px; white-space:pre-wrap;'>{escape(txt)}</pre>"
                )
                self.preview_stack.setCurrentWidget(self.text_view)
            except Exception as e:
                self.info_label.setText(f"Error reading text:\n{e}")
                self.preview_stack.setCurrentWidget(self.info_label)
        elif ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
            pix = QPixmap(path)
            if pix.isNull():
                self.info_label.setText("Cannot load image")
                self.preview_stack.setCurrentWidget(self.info_label)
            else:
                self.image_label.setPixmap(pix)
                self.image_label.resize(pix.size())
                self.preview_stack.setCurrentWidget(self.image_scroll)
        elif ext == ".pdf":
            opened = self._open_file_with_system(path)
            p = Path(path)
            self.text_view.setHtml(
                "<h3 style='margin:0 0 8px 0;'>PDF Report</h3>"
                f"<p><b>File:</b> {escape(p.name)}</p>"
                f"<p><b>Path:</b> {escape(str(p))}</p>"
                f"<p><b>Size:</b> {p.stat().st_size / 1024.0:.1f} KB</p>"
                f"<p><i>{'Opened with system viewer.' if opened else 'Could not auto-open PDF on this system.'}</i></p>"
            )
            self.preview_stack.setCurrentWidget(self.text_view)
        else:
            self.info_label.setText(f"Preview not supported:\n{os.path.basename(path)}")
            self.preview_stack.setCurrentWidget(self.info_label)

    @staticmethod
    def _open_file_with_system(path: str) -> bool:
        try:
            if sys.platform.startswith("linux"):
                subprocess.Popen(["xdg-open", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
            if sys.platform == "darwin":
                subprocess.Popen(["open", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
            if os.name == "nt":
                os.startfile(path)  # type: ignore[attr-defined]
                return True
        except Exception:
            return False
        return False


# -------------------- MAIN WINDOW --------------------
class ADSWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ADS Pipeline Launcher v4.3")
        self.resize(1800, 1100)
        
        self.settings = QSettings("OpenADS", "LauncherV4")
        self.is_running = False 
        self._terminal_proc: Optional[subprocess.Popen] = None
        self._run_watch_timer = QTimer(self)
        self._run_watch_timer.setInterval(1000)
        self._run_watch_timer.timeout.connect(self._check_pipeline_process)
        self._last_exit_code: Optional[int] = None

        self.right_tabs: Optional[QTabWidget] = None
        self.stage_widgets: List[StageUploadWidget] = []
        self.report_files_widget: Optional[ReportFilesWidget] = None
        
        self._setup_ui()
        self._init_defaults()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)

        env_grp = QGroupBox("Environment Configuration")
        env_lay = QFormLayout(env_grp)
        
        self.proj_root_edit = QLineEdit(DEFAULT_PROJECT_ROOT)
        env_lay.addRow("Project Root:", self.proj_root_edit)
        
        self.venv_edit = QLineEdit(DEFAULT_VENV_PATH)
        env_lay.addRow("Virtual Env:", self.venv_edit)
        
        root_layout.addWidget(env_grp)

        splitter = QSplitter(Qt.Horizontal)
        root_layout.addWidget(splitter, 1)

        left_widget = QWidget()
        left_lay = QVBoxLayout(left_widget)
        
        h_mod = QHBoxLayout()
        h_mod.addWidget(QLabel("<b>Modality:</b>"))
        self.mod_combo = QComboBox()
        self.mod_combo.addItems(["DWI", "PWI", "DWI & PWI"])
        self.mod_combo.currentIndexChanged.connect(self._swap_page)
        h_mod.addWidget(self.mod_combo, 1)
        left_lay.addLayout(h_mod)
        
        self.stack = QStackedWidget()
        self.dwi_page = ModalityPage("DWI")
        self.pwi_page = ModalityPage("PWI")
        self.combined_page = CombinedPage()
        self.stack.addWidget(self.dwi_page)
        self.stack.addWidget(self.pwi_page)
        self.stack.addWidget(self.combined_page)
        left_lay.addWidget(self.stack)
        
        self.btn_run = QPushButton("Run Full Pipeline")
        self.btn_run.setMinimumHeight(55)
        self.btn_run.setStyleSheet(
            f"background-color: {COLORS['success']}; color: white; font-weight: bold; "
            "font-size: 13pt; border-radius: 6px;"
        )
        self.btn_run.clicked.connect(self._on_run_clicked)
        left_lay.addWidget(self.btn_run)
        
        self.pbar = QProgressBar()
        self.pbar.setValue(0)
        self.pbar.setTextVisible(False)
        left_lay.addWidget(self.pbar)
        
        splitter.addWidget(left_widget)

        self.right_tabs = QTabWidget()
        self.viewer = NiftiViewer()
        self.right_tabs.addTab(self.viewer, "Preview")
        
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setStyleSheet(
            "background-color: #1e1e1e; color: #00ff00; "
            "font-family: Monospace; font-size: 11pt;"
        )
        self.right_tabs.addTab(self.log_view, "Log / Status")

        self._build_stage_tabs_for_modality("DWI")

        splitter.addWidget(self.right_tabs)
        splitter.setStretchFactor(1, 2)

    def _append_log(self, text: str):
        self.log_view.append(text)

    def _build_stage_tabs_for_modality(self, modality: str):
        while self.right_tabs.count() > 2:
            self.right_tabs.removeTab(2)
        self.stage_widgets.clear()

        if modality in ("DWI", "PWI"):
            page = self.dwi_page if modality == "DWI" else self.pwi_page
            for key, info in PIPELINE_STAGES[modality].items():
                w = StageUploadWidget(
                    modality=modality,
                    stage_key=key,
                    stage_info=info,
                    log_callback=self._append_log,
                    get_subject_path=lambda p=page: p.subject_dir_edit.text(),
                    get_config_path=lambda p=page: p.config_edit.text(),
                    get_gpu=lambda p=page: p.gpu_edit.text(),
                    get_output_root=lambda p=page: p.output_root_edit.text(),
                    parent=self.right_tabs,
                )
                self.stage_widgets.append(w)
                self.right_tabs.addTab(w, info["label"])
            self.report_files_widget = ReportFilesWidget(
                get_subject_id=lambda p=page: p.subject_id_edit.text(),
                get_output_root=lambda p=page: p.output_root_edit.text(),
                parent=self.right_tabs,
            )
        else:
            info = QTextEdit()
            info.setReadOnly(True)
            info.setHtml(
                "<h3>Combined Mode</h3>"
                "<p>Use left panel to select DWI/PWI stages independently.</p>"
                "<p>Run Full Pipeline will execute scripts/run_ads_combined.py.</p>"
            )
            self.right_tabs.addTab(info, "Combined Info")
            self.report_files_widget = ReportFilesWidget(
                get_subject_id=lambda p=self.combined_page: p.subject_id_edit.text(),
                get_output_root=lambda p=self.combined_page: p.output_root_edit.text(),
                parent=self.right_tabs,
            )
        self.right_tabs.addTab(self.report_files_widget, "Report Files")

    def _init_defaults(self):
        ex_dwi = Path(DEFAULT_PROJECT_ROOT) / "assets/examples/dwi/sub-0ab4fbd5"
        if ex_dwi.exists():
            self.dwi_page.subject_dir_edit.setText(str(ex_dwi))
        ex_pwi = Path(DEFAULT_PROJECT_ROOT) / "assets/examples/pwi/sub-0ab4fbd5"
        if ex_pwi.exists():
            self.pwi_page.subject_dir_edit.setText(str(ex_pwi))
        if ex_dwi.exists():
            self.combined_page.dwi_subject_dir_edit.setText(str(ex_dwi))
        if ex_pwi.exists():
            self.combined_page.pwi_subject_dir_edit.setText(str(ex_pwi))
        
        last_gpu = self.settings.value("last_gpu", "1")
        self.dwi_page.gpu_edit.setText(str(last_gpu))
        self.pwi_page.gpu_edit.setText(str(last_gpu))
        self.combined_page.gpu_edit.setText(str(last_gpu))

        self.dwi_page.subject_id_edit.textChanged.connect(self._update_viewer_ctx)
        self.pwi_page.subject_id_edit.textChanged.connect(self._update_viewer_ctx)
        self.combined_page.subject_id_edit.textChanged.connect(self._update_viewer_ctx)

    def closeEvent(self, event):
        if self.mod_combo.currentIndex() == 0:
            curr_page = self.dwi_page
        elif self.mod_combo.currentIndex() == 1:
            curr_page = self.pwi_page
        else:
            curr_page = self.combined_page
        self.settings.setValue("last_gpu", curr_page.gpu_edit.text())
        event.accept()

    def _swap_page(self, idx):
        self.stack.setCurrentIndex(idx)
        modality = "DWI" if idx == 0 else ("PWI" if idx == 1 else "COMBINED")
        self._build_stage_tabs_for_modality(modality)
        self._update_viewer_ctx()

    def _update_viewer_ctx(self):
        idx = self.mod_combo.currentIndex()
        if idx == 0:
            page = self.dwi_page
            root = page.output_root_edit.text()
            sid = page.subject_id_edit.text()
            if root and sid:
                self.viewer.update_context(Path(root) / sid / "DWI", sid)
        elif idx == 1:
            page = self.pwi_page
            root = page.output_root_edit.text()
            sid = page.subject_id_edit.text()
            if root and sid:
                self.viewer.update_context(Path(root) / sid / "PWI", sid)
        else:
            page = self.combined_page
            root = page.output_root_edit.text()
            sid = page.subject_id_edit.text()
            if root and sid:
                self.viewer.update_context(Path(root) / sid / "DWI", sid)
        if self.report_files_widget is not None:
            self.report_files_widget.refresh_files()

    def _on_run_clicked(self):
        if self.is_running:
            reply = QMessageBox.question(
                self, "Reset Pipeline State?",
                "To stop processing, close the terminal window manually.\n\nReset this button to 'Ready'?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self._reset_ui_state()
        else:
            self._launch_pipeline()

    def _reset_ui_state(self):
        self.is_running = False
        self._terminal_proc = None
        self._last_exit_code = None
        if self._run_watch_timer.isActive():
            self._run_watch_timer.stop()
        self.btn_run.setText("Run Full Pipeline")
        self.btn_run.setStyleSheet(
            f"background-color: {COLORS['success']}; color: white; font-weight: bold; "
            "font-size: 13pt; border-radius: 6px;"
        )
        self.pbar.setValue(0)
        self.log_view.append("<b>[RESET]</b> Interface ready.")

    def _check_pipeline_process(self):
        if not self._terminal_proc:
            return
        if self._terminal_proc.poll() is None:
            return

        self._last_exit_code = self._terminal_proc.returncode
        self._run_watch_timer.stop()
        self._terminal_proc = None
        # Small delay to align UI state with terminal closing animation.
        QTimer.singleShot(2500, self._finalize_pipeline_status)

    def _finalize_pipeline_status(self):
        if self._last_exit_code is None:
            return
        exit_code = self._last_exit_code
        self._last_exit_code = None
        self.is_running = False

        if exit_code == 0:
            self.btn_run.setText("Completed (Click to Run Again)")
            self.btn_run.setStyleSheet(
                f"background-color: {COLORS['success']}; color: white; font-weight: bold; "
                "font-size: 13pt; border-radius: 6px;"
            )
            self.pbar.setValue(100)
            self.log_view.append("<b>[DONE]</b> Pipeline completed successfully.")
        else:
            self.btn_run.setText("Failed (Click to Run Again)")
            self.btn_run.setStyleSheet(
                f"background-color: {COLORS['danger']}; color: white; font-weight: bold; "
                "font-size: 13pt; border-radius: 6px;"
            )
            self.pbar.setValue(0)
            self.log_view.append(f"<b>[FAILED]</b> Pipeline exited with code {exit_code}.")

    def _launch_pipeline(self):
        proj_root = self.proj_root_edit.text()
        venv = self.venv_edit.text()
        idx = self.mod_combo.currentIndex()
        launch_label = ""
        subject_label = ""
        stages_label = ""

        if idx in (0, 1):
            is_dwi = (idx == 0)
            page = self.dwi_page if is_dwi else self.pwi_page
            subj_path = page.subject_dir_edit.text()
            if not subj_path:
                QMessageBox.warning(self, "Error", "Please select a Subject Folder first.")
                return
            script_name = "run_dwi.sh" if is_dwi else "run_pwi.sh"
            config_path = page.config_edit.text()
            if not config_path.startswith("/"):
                config_path = f"{proj_root}/{config_path}"
            gpu = page.gpu_edit.text()
            out_root = page.output_root_edit.text().strip()
            stages = [k for k, cb in page.stage_checks.items() if cb.isChecked()]
            stages_str = ",".join(stages)

            cmd_parts = [
                f"cd \"{proj_root}\"",
                f"source \"{venv}/bin/activate\"",
                f"bash scripts/{script_name} \"{subj_path}\" --config \"{config_path}\" --gpu {gpu}",
            ]
            if out_root:
                cmd_parts[-1] += f" --output-root \"{out_root}\""
            if stages_str:
                cmd_parts[-1] += f" --stages {stages_str}"
            cmd = " && ".join(cmd_parts) + " && echo '' && echo 'Pipeline completed successfully!' && sleep 3 && exit"
            launch_label = script_name
            subject_label = Path(subj_path).name
            stages_label = stages_str
        else:
            page = self.combined_page
            dwi_path = page.dwi_subject_dir_edit.text()
            pwi_path = page.pwi_subject_dir_edit.text()
            if not dwi_path or not pwi_path:
                QMessageBox.warning(self, "Error", "Please select both DWI and PWI Subject Folders.")
                return
            dwi_cfg = page.dwi_config_edit.text()
            pwi_cfg = page.pwi_config_edit.text()
            if not dwi_cfg.startswith("/"):
                dwi_cfg = f"{proj_root}/{dwi_cfg}"
            if not pwi_cfg.startswith("/"):
                pwi_cfg = f"{proj_root}/{pwi_cfg}"
            gpu = page.gpu_edit.text()
            out_root = page.output_root_edit.text().strip()
            dwi_stages_list = page.selected_dwi_stages()
            pwi_stages_list = page.selected_pwi_stages()

            if not dwi_stages_list and not pwi_stages_list:
                QMessageBox.warning(self, "Error", "Please select at least one stage for DWI or PWI.")
                return

            run_steps = [f"cd \"{proj_root}\"", f"source \"{venv}/bin/activate\""]
            if dwi_stages_list:
                dwi_cmd = f"bash scripts/run_dwi.sh \"{dwi_path}\" --config \"{dwi_cfg}\" --gpu {gpu}"
                if out_root:
                    dwi_cmd += f" --output-root \"{out_root}\""
                dwi_cmd += f" --stages {','.join(dwi_stages_list)}"
                run_steps.append(dwi_cmd)
            if pwi_stages_list:
                pwi_cmd = f"bash scripts/run_pwi.sh \"{pwi_path}\" --config \"{pwi_cfg}\" --gpu {gpu}"
                if out_root:
                    pwi_cmd += f" --output-root \"{out_root}\""
                pwi_cmd += f" --stages {','.join(pwi_stages_list)}"
                run_steps.append(pwi_cmd)
            cmd = " && ".join(run_steps) + " && echo '' && echo 'Combined pipeline completed successfully!' && sleep 3 && exit"
            launch_label = "run_dwi.sh + run_pwi.sh"
            subject_label = f"DWI={Path(dwi_path).name}, PWI={Path(pwi_path).name}"
            stages_label = (
                f"DWI[{','.join(dwi_stages_list) or 'yaml'}], "
                f"PWI[{','.join(pwi_stages_list) or 'yaml'}]"
            )
        
        term_cmd = []
        if shutil.which("gnome-terminal"):
            term_cmd = ["gnome-terminal", "--geometry=60x15", "--", "bash", "-c", cmd]
        elif shutil.which("konsole"):
            term_cmd = ["konsole", "--geometry", "600x300", "-e", "bash", "-c", cmd]
        elif shutil.which("xterm"):
            term_cmd = ["xterm", "-geometry", "60x15", "-e", "bash", "-c", cmd]
        else:
            QMessageBox.critical(self, "Error", "No compatible terminal found.")
            return

        try:
            self._terminal_proc = subprocess.Popen(term_cmd)
            self.is_running = True
            if not self._run_watch_timer.isActive():
                self._run_watch_timer.start()
            self.btn_run.setText("Pipeline Running... (Click to Reset)")
            self.btn_run.setStyleSheet(
                f"background-color: {COLORS['danger']}; color: white; font-weight: bold; "
                "font-size: 13pt; border-radius: 6px;"
            )
            self.pbar.setValue(50)
            self.log_view.append(f"<b>[LAUNCHED]</b> {launch_label}")
            self.log_view.append(f"Subject: {subject_label}")
            self.log_view.append(f"Stages: {stages_label}")
            
        except Exception as e:
            QMessageBox.critical(self, "Launch Failed", str(e))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    font = QFont()
    font.setPointSize(11)
    app.setFont(font)
    
    w = ADSWindow()
    w.show()
    sys.exit(app.exec_())
