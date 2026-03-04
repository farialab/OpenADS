import os
import sys
import torch
import numpy as np
import surfa as sf
import scipy
import torch.nn as nn
from pathlib import Path
from contextlib import contextmanager

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_DIR = PACKAGE_ROOT / "assets" / "models"

# Context manager to suppress stdout and stderr
@contextmanager
def suppress_stdout_stderr():
    """
    A context manager that redirects stdout and stderr to os.devnull (suppresses output).
    """
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            
# Define the ConvBlock class
class ConvBlock(nn.Module):
    """
    Convolutional block followed by LeakyReLU activation for the U-Net architecture.
    """
    def __init__(self, ndims, in_channels, out_channels, stride=1, activation='leaky'):
        super().__init__()
        Conv = getattr(nn, f'Conv{ndims}d')
        self.conv = Conv(in_channels, out_channels, 3, stride, 1)
        if activation == 'leaky':
            self.activation = nn.LeakyReLU(0.2)
        elif activation is None:
            self.activation = None
        else:
            raise ValueError(f'Unknown activation: {activation}')

    def forward(self, x):
        out = self.conv(x)
        if self.activation is not None:
            out = self.activation(out)
        return out

# Define the StripModel class
class StripModel(nn.Module):
    """
    U-Net architecture used by SynthStrip for brain extraction.
    """
    def __init__(self,
                 nb_features=16,
                 nb_levels=7,
                 feat_mult=2,
                 max_features=64,
                 nb_conv_per_level=2,
                 max_pool=2,
                 return_mask=False):

        super().__init__()

        # Dimensionality
        ndims = 3

        # Build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('Must provide nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            feats = np.clip(feats, 1, max_features)
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level)
            ]
        elif nb_levels is not None:
            raise ValueError('Cannot use nb_levels if nb_features is not an integer')

        # Extract any surplus (full resolution) decoder convolutions
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

        if isinstance(max_pool, int):
            max_pool = [max_pool] * self.nb_levels

        # Cache downsampling / upsampling operations
        MaxPooling = getattr(nn, f'MaxPool{ndims}d')
        self.pooling = nn.ModuleList([MaxPooling(s) for s in max_pool])
        self.upsampling = nn.ModuleList([nn.Upsample(scale_factor=s, mode='nearest') for s in max_pool])

        # Configure encoder (down-sampling path)
        prev_nf = 1
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.encoder.append(convs)
            encoder_nfs.append(prev_nf)

        # Configure decoder (up-sampling path)
        encoder_nfs = np.flip(encoder_nfs)
        self.decoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.decoder.append(convs)
            if level < (self.nb_levels - 1):
                prev_nf += encoder_nfs[level]

        # Handle any remaining convolutions
        self.remaining = nn.ModuleList()
        for nf in final_convs:
            self.remaining.append(ConvBlock(ndims, prev_nf, nf))
            prev_nf = nf

        # Final convolutions
        if return_mask:
            self.remaining.append(ConvBlock(ndims, prev_nf, 2, activation=None))
            self.remaining.append(nn.Softmax(dim=1))
        else:
            self.remaining.append(ConvBlock(ndims, prev_nf, 1, activation=None))

    def forward(self, x):
        # Encoder forward pass
        x_history = [x]
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                x = conv(x)
            x_history.append(x)
            x = self.pooling[level](x)

        # Decoder forward pass with upsampling and concatenation
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                x = conv(x)
            if level < (self.nb_levels - 1):
                x = self.upsampling[level](x)
                x = torch.cat([x, x_history.pop()], dim=1)

        # Remaining convolutions at full resolution
        for conv in self.remaining:
            x = conv(x)

        return x

# Define the function to generate the brain mask
def generate_brain_mask_with_synthstrip(input_image_path, output_mask_path, use_gpu=True, no_csf=False, model_path=None):
    """
    Generate a brain mask from a DWI image using SynthStrip.

    Parameters:
    - input_image_path: str, path to the input DWI image.
    - output_mask_path: str, path where the output mask will be saved.
    - use_gpu: bool, whether to use GPU for processing.
    - no_csf: bool, whether to exclude CSF from the brain boundary.
    - model_path: str, path to the SynthStrip model file.
    """
    # Check if the input file exists
    if not os.path.exists(input_image_path):
        raise FileNotFoundError(f"Input file {input_image_path} not found")

    # Set up device
    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
    else:
        device = torch.device('cpu')

    # Load the model
    model = StripModel()
    model.to(device)
    model.eval()

    # Load model weights
    if model_path is not None:
        modelfile = Path(model_path)
    else:
        version = "1"
        filename = f"synthstrip.nocsf.{version}.pt" if no_csf else f"synthstrip.{version}.pt"
        modelfile = DEFAULT_MODEL_DIR / filename
    if not modelfile.exists():
        raise FileNotFoundError(f"Model file {modelfile} not found. Please provide the correct path.")
    checkpoint = torch.load(str(modelfile), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Load input volume
    image = sf.load_volume(input_image_path)

    # Process each frame
    mask_frames = []
    for f in range(image.nframes):
        frame = image.new(image.framed_data[..., f])
        frame = frame.astype(np.float32)

        # Conform image
        conformed = frame.conform(voxsize=1.0, dtype='float32', method='nearest', orientation='LIA').crop_to_bbox()
        target_shape = np.clip(np.ceil(np.array(conformed.shape[:3]) / 64).astype(int) * 64, 192, 320)
        conformed = conformed.reshape(target_shape)

        # Normalize intensities
        conformed -= conformed.min()
        conformed = (conformed / conformed.percentile(99)).clip(0, 1)

        # Predict the surface distance transform
        with torch.no_grad():
            input_tensor = torch.from_numpy(conformed.data[np.newaxis, np.newaxis]).to(device)
            sdt = model(input_tensor).cpu().numpy().squeeze()

        # Generate the mask
        mask_frame = (sdt < 0).astype(np.uint8)

        # Resample back to original space
        sdt_volume = conformed.new(sdt)
        sdt_resampled = sdt_volume.resample_like(frame, fill=100)
        mask_frame_resampled = (sdt_resampled.data < 0).astype(np.uint8)
        mask_frame_resampled = sf.Volume(mask_frame_resampled, frame.geom)

        # Find the largest connected component
        mask_frame_resampled = mask_frame_resampled.connected_component_mask(k=1, fill=True)

        mask_frames.append(mask_frame_resampled.data)

    # Combine frames
    mask_data = np.stack(mask_frames, axis=-1)

    # Create a new volume for the mask
    mask_volume = image.new(mask_data)

    # Save the mask
    mask_volume.save(output_mask_path)

    return mask_volume
