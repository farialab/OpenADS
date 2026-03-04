import os
import yaml
from pathlib import Path
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple, Union
import argparse

def get_project_paths(base_dir: Path = None) -> Dict[str, Path]:
    """Get project directory paths"""
    if base_dir is None:
        # Determine project root directory (2 levels up from this file)
        #base_dir = Path(__file__).parent.parent.parent
        base_dir = Path(__file__).resolve().parents[3]

    return {
        'project': base_dir,
        'template': base_dir / 'assets' / 'atlases',
        'models': base_dir / 'assets' / 'models',
        'aa_models': base_dir / 'assets' / 'models' / 'AA_models'
    }

def load_config(config_path):
    """Load configuration from YAML file"""
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# def load_config(config_path: str) -> Dict[str, Any]:
#     """Load YAML configuration file"""
#     with open(config_path, 'r') as f:
#         config = yaml.safe_load(f)
    
#     # Resolve environment variables and template strings
#     config = _resolve_config_vars(config)
    
#     return config

def _resolve_config_vars(config: Dict[str, Any], env_vars: Dict[str, str] = None) -> Dict[str, Any]:
    """Resolve environment variables and template strings in configuration"""
    if env_vars is None:
        env_vars = {
            'DATA_DIR': str(get_project_paths()['project'] / 'data')
        }
    
    if isinstance(config, dict):
        for key, value in config.items():
            if isinstance(value, (dict, list)):
                config[key] = _resolve_config_vars(value, env_vars)
            elif isinstance(value, str):
                # Replace environment variables
                for env_key, env_value in env_vars.items():
                    value = value.replace(f"${{{env_key}}}", env_value)
                
                # Replace template strings
                if "${" in value and "}" in value:
                    for config_key in _find_template_vars(value):
                        config_value = _get_nested_config_value(config, config_key)
                        if config_value is not None:
                            value = value.replace(f"${{{config_key}}}", str(config_value))
                
                config[key] = value
    elif isinstance(config, list):
        return [_resolve_config_vars(item, env_vars) for item in config]
    
    return config

def _find_template_vars(value: str) -> List[str]:
    """Find template variables in a string"""
    result = []
    start = 0
    while True:
        start = value.find("${", start)
        if start == -1:
            break
        end = value.find("}", start)
        if end == -1:
            break
        result.append(value[start+2:end])
        start = end + 1
    return result

def _get_nested_config_value(config: Dict[str, Any], key_path: str) -> Any:
    """Get a nested configuration value using dot notation"""
    keys = key_path.split('.')
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return None
    return value

def load_excel_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from Excel file (legacy format)"""
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        config_df = pd.read_excel(config_path)
    
    # Convert dataframe to dictionary
    config_dict = get_config_dict(config_df)
    
    # Add paths
    paths = get_project_paths()
    config_dict.update({
        'template_dir': str(paths['template']),
        'aa_models_dir': str(paths['aa_models']),
        'lesion_model_path': str(paths['models'] / f"{config_dict['lesion_model_name']}.pth")
    })
    
    return config_dict

def get_config_dict(config_df: pd.DataFrame) -> Dict[str, Any]:
    """Get configuration dictionary from dataframe (legacy format)"""
    config_dict = {
        'lesion_model_name': config_df['selected_option'][1],
        'n_channel': 2 if 'CH2' in config_df['selected_option'][1] else 3,
        'bvalue': int(config_df['selected_option'][2]),
        'save_mni': config_df['selected_option'][3],
        'mni_spec': config_df['selected_option'][4],
        'nonlinear_registration': config_df['selected_option'][5],
        'generate_brainmask': config_df['selected_option'][6],
        'brainmask_option': config_df['selected_option'][7],
        'generate_report': config_df['selected_option'][8],
        'generate_result_png': config_df['selected_option'][9],
        'generate_radiological_report': config_df['selected_option'][10],
        'radiological_report_model': config_df['selected_option'][11],
        'generate_report_exp': config_df['selected_option'][12],
        'nonlinear_hydrocephalus': config_df['selected_option'][13],
        'hyperperfusion': config_df['selected_option'][14],
        'pwi_motion_correction': config_df['selected_option'][15],
        'generate_ttp': config_df['selected_option'][16],
        'synthstrip_model': config_df['selected_option'][17]
    }
    return config_dict

def get_subject_list(config_df: pd.DataFrame) -> List[str]:
    """Get list of subject directories from configuration dataframe"""
    with open(config_df['selected_option'][0], 'r') as f:
        subject_dirs = f.readlines()
    
    return [dir_path.strip() for dir_path in subject_dirs]


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Acute Stroke Detection - Inference Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--input", "-i", 
        type=str, 
        required=True,
        help="Path to subject directory"
    )

    parser.add_argument(
        "--only-report",
        action="store_true",
        help="Only run the report generation step"
    )
    
    parser.add_argument(
        "--config", "-c", 
        type=str, 
        default=str(Path(__file__).resolve().parents[3] / "configs" / "defaults.yaml"),
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--model", "-m", 
        type=str, 
        default=None,
        choices=["DAGMNet_CH3", "DAGMNet_CH2", "UNet_CH3", "UNet_CH2", "FCN_CH3", "FCN_CH2"],
        help="Model type to use (overrides config)"
    )
    
    parser.add_argument(
        "--template-dir", "-t",
        type=str,
        default=None,
        help="Directory containing template files (overrides config)"
    )
    
    parser.add_argument(
        "--output-dir", "-o", 
        type=str, 
        default=None,
        help="Output directory (defaults to input directory)"
    )
    
    parser.add_argument(
        "--gpu", "-g", 
        type=str, 
        default=None,
        help="GPU ID to use (empty for CPU)"
    )
    
    parser.add_argument(
        "--bvalue", "-b", 
        type=int, 
        default=None,
        help="B-value for ADC calculation"
    )
    
    parser.add_argument(
        "--use-b0",
        action="store_true",
        help="Use B0 image if available (default: False)"
    )
    
    parser.add_argument(
        "--save-mni", 
        action="store_true",
        help="Save results in MNI space"
    )
    
    parser.add_argument(
        "--generate-report", 
        action="store_true",
        help="Generate lesion report"
    )
    
    parser.add_argument(
        "--generate-visualization", 
        action="store_true",
        help="Generate result visualization"
    )
    
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


import os
import yaml
import re
from pathlib import Path
from typing import Dict, Any, Union

class ConfigManager:
    """Elegant configuration loader with inheritance and variable expansion."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root

    def load_pipeline_config(self, pipeline_config_path: Union[str, Path]) -> Dict[str, Any]:
        """Loads a pipeline config and merges it with defaults."""
        pipeline_config_path = Path(pipeline_config_path)
        config_dir = pipeline_config_path.parent
        
        # 1. Load Defaults
        defaults = self._load_yaml(config_dir / "defaults.yaml")
        
        # 2. Load Pipeline Config
        pipeline = self._load_yaml(pipeline_config_path)
        
        # 3. Merge (Pipeline overrides Defaults)
        final_config = self._deep_update(defaults, pipeline)
        
        # 4. Expand Variables
        final_config = self._expand_variables(final_config)
        
        return final_config

    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        with open(path, 'r') as f:
            return yaml.safe_load(f) or {}

    def _deep_update(self, base: Dict, update: Dict) -> Dict:
        for k, v in update.items():
            if isinstance(v, dict) and k in base and isinstance(base[k], dict):
                self._deep_update(base[k], v)
            else:
                base[k] = v
        return base

    def _expand_variables(self, config: Any) -> Any:
        """Recursively expand ${PROJECT_ROOT} and other variables."""
        if isinstance(config, dict):
            return {k: self._expand_variables(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._expand_variables(i) for i in config]
        elif isinstance(config, str):
            if "${PROJECT_ROOT}" in config:
                config = config.replace("${PROJECT_ROOT}", str(self.project_root))
            # Can add more substitutions here if needed
            return config
        return config

# Helper for scripts
def load_config(config_path: Path, project_root: Path = None) -> Dict[str, Any]:
    if project_root is None:
        project_root = Path(__file__).resolve().parents[3]
    return ConfigManager(project_root).load_pipeline_config(config_path)

def find_subject_files(input_dir, subject_id):
    """Find subject files with flexible extensions"""
    import os
    
    files = {}
    
    # Try to find DWI file
    dwi_path = None
    for ext in ['.nii.gz', '.nii']:
        test_path = os.path.join(input_dir, f"{subject_id}_DWI{ext}")
        if os.path.exists(test_path):
            dwi_path = test_path
            break
    files['dwi'] = dwi_path
    
    # Try to find ADC file
    adc_path = None
    for ext in ['.nii.gz', '.nii']:
        test_path = os.path.join(input_dir, f"{subject_id}_ADC{ext}")
        if os.path.exists(test_path):
            adc_path = test_path
            break
    files['adc'] = adc_path
    
    # Try to find B0 file
    b0_path = None
    for ext in ['.nii.gz', '.nii']:
        test_path = os.path.join(input_dir, f"{subject_id}_b0{ext}")
        if os.path.exists(test_path):
            b0_path = test_path
            break
    files['b0'] = b0_path
    
    # Try to find brain mask file
    mask_path = None
    for ext in ['.nii.gz', '.nii']:
        test_path = os.path.join(input_dir, f"{subject_id}_brain_mask{ext}")
        if os.path.exists(test_path):
            mask_path = test_path
            break
    files['mask'] = mask_path
    
    return files
