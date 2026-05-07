"""Readout module for spot detection and intensity extraction."""

from .spot_detection import (
    preprocess_image_in_memory,  # Backward compatibility alias
    feature_gaussian_tophat,
    feature_gaussian_dog,
    feature_tophat,
    feature_dog,
    blob_log_detection,
    get_spot_coordinates,
    get_coordinates_for_tile,
)

from .image_blocks import block_starts

from .intensity_readout import (
    read_intensity_raw,
    read_intensity_tophat,
    read_intensity_integrated,
    read_intensity_gaussian_fit,
    get_intensity_df_for_tile,
    extract_intensity_for_tile,
)

__all__ = [
    # Backward compatibility
    'preprocess_image_in_memory',
    # Feature extraction
    'feature_gaussian_tophat',
    'feature_gaussian_dog',
    'feature_tophat',
    'feature_dog',
    # Spot detection
    'blob_log_detection',
    'get_spot_coordinates',
    'get_coordinates_for_tile',
    # Intensity reading
    'read_intensity_raw',
    'read_intensity_tophat',
    'read_intensity_integrated',
    'read_intensity_gaussian_fit',
    'get_intensity_df_for_tile',
    'extract_intensity_for_tile',
    # Block utilities
    'block_starts',
]

