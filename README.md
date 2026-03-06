# OpenADS

OpenADS is an end to end medical imaging pipeline for stroke analysis, covering preprocessing, registration, segmentation, and report generation for DWI and PWI workflows.

![OpenADS UI](server/ui.png)

## What OpenADS does

OpenADS runs standardized pipelines that take a single subject folder and produce:
- Preprocessed volumes
- Brain masks and skull stripped images
- MNI registration outputs
- Model inference masks
- Postprocessed results and metrics
- Human readable text reports and PNG visualizations

Supported workflows:
- DWI only pipeline
- PWI only pipeline
- Combined DWI plus PWI pipeline for matched subjects

## Quickstart

Run commands from the OpenADS project root.

Create and activate your environment:

```bash
python -m venv openads
source openads/bin/activate

python -m pip install -U pip
python -m pip install torch==2.5.1 numpy==2.0.1 antspyx==0.5.4 pandas==2.2.3 matplotlib==3.9.2 scikit-image==0.24.0 scipy==1.13.1 nibabel==5.3.2 tqdm==4.67.1 scikit-learn==1.7.2 surfa==0.6.3
```

Run a full DWI pipeline on an example subject:

```bash
./scripts/run_dwi.sh assets/examples/dwi/sub-02e8eb42 --all --gpu 1
```

Run a full PWI pipeline on an example subject:

```bash
./scripts/run_pwi.sh assets/examples/pwi/sub-02e8eb42 --all --gpu 1
```

Run the combined pipeline:

```bash
./scripts/run_combined.sh \
  --dwi assets/examples/dwi/sub-02e8eb42 \
  --pwi assets/examples/pwi/sub-02e8eb42 \
  --all \
  --gpu 1
```



## Installation

OpenADS can be run as:

* Local scripts and Python entrypoints
* A package CLI via `python -m ads.cli`
* A web API and GUI launcher via Uvicorn
* Docker images (CPU and GPU)

### Option A: Install from source (recommended)

Clone the repository and create a dedicated virtual environment:

```bash
git clone https://github.com/farialab/OpenADS.git
cd OpenADS

python -m venv openads
source openads/bin/activate

python -m pip install -U pip
python -m pip install .
```
Sanity check:

```bash
ads --help
```
### Option B: Editable install (for development)

Use editable mode if you are making changes to the code:
```bash
git clone https://github.com/farialab/OpenADS.git
cd OpenADS

python -m venv openads
source openads/bin/activate

python -m pip install -U pip
python -m pip install -e .
```
Developer sanity checks:

```bash
python -c "import ads; print('ads import OK')"
ads --help
```

## Input data expectations

* Provide a single subject folder as the subject path, for example `.../sub-xxxx`.
* Do not pass a dataset root to `--subject-path`. 

## Outputs and output root

By default, output is written under `output/{subject_id}/DWI/...` and `output/{subject_id}/PWI/...`. 

You can override the output base with `--output-root` in:

* `scripts/run_dwi.sh`, `scripts/run_pwi.sh`, `scripts/run_combined.sh`
* `scripts/run_ads_dwi.py`, `scripts/run_ads_pwi.py`, `scripts/run_ads_combined.py` 

Cleanup and keep behavior is controlled by `configs/keep_files.yaml`. 

## Example output tree

This is a real example tree for `output/sub-05a971ae` (trimmed for display):

```
output/sub-05a971ae
├── DWI
│   ├── preprocess
│   ├── registration
│   ├── reporting
│   └── segment
└── PWI
    ├── preprocess
    ├── registration
    ├── reporting
    └── segmentation
```

## Running OpenADS

OpenADS provides multiple entry options depending on how you want to integrate it.

### Option 1: Local shell wrappers

DWI single subject:

```bash
./scripts/run_dwi.sh assets/examples/dwi/sub-02e8eb42 --all --gpu 1
./scripts/run_dwi.sh assets/examples/dwi/sub-02e8eb42 --stages prepdata,gen_mask,skull_strip,registration,inference,report --gpu 1
./scripts/run_dwi.sh /data/dwi/sub-0037761d --stages prepdata,gen_mask,skull_strip,registration,inference,report --gpu 1 --output-root /output
```

PWI single subject:

```bash
./scripts/run_pwi.sh assets/examples/pwi/sub-02e8eb42 --all --gpu 1
./scripts/run_pwi.sh assets/examples/pwi/sub-02e8eb42 --stages prepdata,gen_mask,skull_strip,gen_ttp,registration,ttpadc_coreg,inference,report --gpu 1
./scripts/run_pwi.sh /data/pwi/sub-0037761d --stages prepdata,gen_mask,skull_strip,gen_ttp,registration,ttpadc_coreg,inference,report --gpu 1 --output-root /output
```

### Option 2: Combined wrapper

Combined DWI plus PWI:

```bash
./scripts/run_combined.sh \
  --dwi assets/examples/dwi/sub-02e8eb42 \
  --pwi assets/examples/pwi/sub-02e8eb42 \
  --all \
  --gpu 1
```

Custom stage selection:

```bash
./scripts/run_combined.sh \
  --dwi assets/examples/dwi/sub-02e8eb42 \
  --pwi assets/examples/pwi/sub-02e8eb42 \
  --dwi-stages prepdata,gen_mask,skull_strip,registration,inference,report \
  --pwi-stages prepdata,gen_mask,skull_strip,gen_ttp,registration,ttpadc_coreg,inference,report \
  --gpu 1 \
  --output-root /data/openads_output
```

Run only PWI stages in the combined wrapper:

```bash
./scripts/run_combined.sh \
  --dwi assets/examples/dwi/sub-02e8eb42 \
  --pwi assets/examples/pwi/sub-02e8eb42 \
  --pwi-stages prepdata,gen_mask,skull_strip,gen_ttp,registration,ttpadc_coreg,inference,report \
  --gpu 1
```

Combined wrapper behavior:

* Executes DWI first, then PWI.
* With `--all` runs both full pipelines.
* Without `--all`, DWI runs only when `--dwi-stages` is provided and PWI runs only when `--pwi-stages` is provided.
* If both stage lists are empty, it exits with an error.
* `--output-root` sends both modalities to the same output base.
* `--no-mask-copy` is accepted for compatibility but is a no op in this wrapper. 

### Option 3: Batch wrappers

DWI batch:

```bash
./scripts/batch_run_dwi.sh --subjects-root assets/examples/dwi --all --gpu 1
./scripts/batch_run_dwi.sh --subjects-root assets/examples/dwi --parallel --max-jobs 2 --all --gpu 1
./scripts/batch_run_dwi.sh --subjects-file /abs/path/dwi_subjects.txt --stages prepdata,gen_mask,skull_strip,registration,inference,report --gpu 1
```

PWI batch:

```bash
./scripts/batch_run_pwi.sh --subjects-root assets/examples/pwi --all --gpu 1
./scripts/batch_run_pwi.sh --subjects-root assets/examples/pwi --parallel --max-jobs 2 --all --gpu 1
./scripts/batch_run_pwi.sh --subjects-file /abs/path/pwi_subjects.txt --stages prepdata,gen_mask,skull_strip,gen_ttp,registration,ttpadc_coreg,inference,report --gpu 1
```

Combined batch:

```bash
./scripts/batch_run_combined.sh \
  --dwi-root assets/examples/dwi \
  --pwi-root assets/examples/pwi \
  --all \
  --gpu 1
```



### Option 4: Python entrypoints

```bash
python scripts/run_ads_dwi.py --subject-path assets/examples/dwi/sub-02e8eb42 --config configs/dwi_pipeline.yaml --all --gpu 1
python scripts/run_ads_pwi.py --subject-path assets/examples/pwi/sub-02e8eb42 --config configs/pwi_pipeline.yaml --all --gpu 1
python scripts/run_ads_combined.py --dwi-subject-path assets/examples/dwi/sub-02e8eb42 --pwi-subject-path assets/examples/pwi/sub-02e8eb42 --all --gpu 1
```

Custom output root:

```bash
python scripts/run_ads_dwi.py --subject-path /data/dwi/sub-0037761d --config configs/dwi_pipeline.yaml --stages prepdata,gen_mask,skull_strip,registration,inference,report --gpu 1 --output-root /data/openads_output
```



### Option 5: Package CLI

```bash
PYTHONPATH=src python -m ads.cli dwi --subject-path assets/examples/dwi/sub-02e8eb42 --all --gpu 1
PYTHONPATH=src python -m ads.cli pwi --subject-path assets/examples/pwi/sub-02e8eb42 --all --gpu 1
```



## Web API and GUI

Launch the server:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

Open:

* `http://127.0.0.1:8000/`
* `http://127.0.0.1:8000/gui_launcher.html`

Notes:

* The web launcher forwards user provided subject paths and output root to backend scripts.
* Report file listing and download in the web launcher uses the selected output root. 

## Docker

Build images:

```bash
docker build -f docker/Dockerfile.cpu -t openads:cpu .
docker build -f docker/Dockerfile.gpu -t openads:gpu .
```

Entrypoint commands:

* `ads api`
* `ads dwi ...`
* `ads pwi ...`
* `ads combined ...`
* `ads cli ...`
* `ads gui` 

Run API:

```bash
docker run --rm -it \
  --user "$(id -u):$(id -g)" \
  -p 8000:8000 \
  -v "$(pwd)/output:/app/output" \
  openads:cpu api
```

Run pipelines:

```bash
docker run --rm -it --user "$(id -u):$(id -g)" -v "$(pwd)/output:/app/output" openads:cpu dwi --subject-path /app/assets/examples/dwi/sub-02e8eb42 --all --gpu 1
docker run --rm -it --user "$(id -u):$(id -g)" -v "$(pwd)/output:/app/output" openads:cpu pwi --subject-path /app/assets/examples/pwi/sub-02e8eb42 --all --gpu 1
docker run --rm -it --user "$(id -u):$(id -g)" -v "$(pwd)/output:/app/output" openads:cpu combined --dwi-subject-path /app/assets/examples/dwi/sub-02e8eb42 --pwi-subject-path /app/assets/examples/pwi/sub-02e8eb42 --all --gpu 1
```

GPU containers:

* Add `--gpus all`
* Use `openads:gpu` 

## Pipeline stages

OpenADS pipelines are composed of named stages. These names appear in `--stages` arguments and represent a stable contract for orchestration. 

### DWI stages

1. `prepdata`
   Loads raw DWI and B0, computes ADC if needed, fixes volume order if required, and saves preprocessed files. 

2. `gen_mask`
   Generates a brain mask (SynthStrip). 

3. `skull_strip`
   Applies the mask to DWI, B0, and ADC and saves skull stripped images. 

4. `registration`
   Registers to MNI152 (affine plus SyN), applies transforms, and saves normalized outputs and transform files. 

5. `inference`
   Runs stroke lesion segmentation (DAGMNet) in MNI space. 

6. `postprocessing`
   Creates additional space variants, transforms back to native space, and computes metrics. 

7. `report`
   Computes atlas overlaps and QFV features and generates text reports and PNG visualizations. 

### PWI stages

1. `prepdata`
   Loads PWI 4D and related volumes, preserves JSON sidecar metadata, and saves standardized outputs. 

2. `gen_mask`
   Brain masking (reuses DWI method). 

3. `skull_strip`
   Skull stripping (reuses DWI method). 

4. `gen_ttp`
   Computes time to peak maps from PWI 4D and saves TTP. 

5. `registration`
   Registers to MNI152 and normalizes DWI, ADC, and TTP variants. 

6. `ttpadc_coreg`
   Coregisters TTP into ADC or DWI space and saves transform matrices. 

7. `inference`
   Runs hypoperfusion segmentation and outputs HP masks. 

8. `postprocessing`
   Transforms back to native space and computes metrics. 

9. `report`
   Generates atlas overlap features, QFV outputs, mismatch statements, and PNG visualizations. 

## Architecture overview

OpenADS follows a clean architecture style with unidirectional dependencies: entrypoints call pipelines, pipelines orchestrate domain and services, services use adapters, and the domain layer remains dependency free. 

High level layers:

* Entrypoints layer: CLI, argument parsing, logging setup
* Pipelines layer: stage orchestration and workflow control
* Domain layer: pure data objects
* Services layer: business logic and algorithms
* Adapters layer: I O, filesystem, external systems 

## Configuration

Pipeline behavior is controlled by configuration files such as:

* `configs/dwi_pipeline.yaml`
* `configs/pwi_pipeline.yaml`
* `configs/keep_files.yaml` 

## Troubleshooting

Common gotchas:

* Pass a single subject folder to `--subject-path`, not a dataset root. 
* Use `--output-root` if you want outputs outside the default `output/` tree. 
* If running Docker and you want outputs on the host, mount a host directory into the container and set `--output-root` to that mount point if needed. 

## Contributing

Contributions are welcome.

## License

Add your license text or reference here.

## Citation

