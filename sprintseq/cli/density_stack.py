"""Composite density stack builder with ImageJ-compatible thermal LUT.

Reads per-gene density TIFFs, applies Gaussian blur, and writes a single
multi-page TIFF with embedded thermal colormap and display range metadata.

Usage:
    sprintseq density-stack --run-id <run_id> --gene-file genes.txt
    sprintseq density-stack --run-id <run_id> --all --threshold 0.95
"""

import os
import argparse
import logging
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from tifffile import imread, imwrite

logger = logging.getLogger(__name__)

BASE_DEST_DIRECTORY = r'\\10.10.10.1\NAS Processed Images'
DEFAULT_QUALITY = 20
DEFAULT_THRESHOLD = 0.99  # Q20
DEFAULT_SIGMA = 0.7
DEFAULT_DISPLAY_MIN = 1.0
DEFAULT_DISPLAY_MAX = 10.0

# fmt: off
_THERMAL_R = [
    0, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70,
    70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70,
    70, 70, 70, 70, 70, 70, 70, 69, 69, 68, 67, 66, 64, 63, 61, 59,
    57, 55, 53, 51, 48, 45, 43, 40, 37, 34, 31, 28, 26, 23, 20, 17,
    14, 12, 9, 6, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 1, 4, 7, 11, 16, 21, 27, 32, 39, 45, 52, 60, 68,
    76, 84, 92, 101, 109, 117, 126, 134, 143, 152, 160, 168, 176, 184, 192, 199,
    206, 213, 219, 225, 231, 236, 240, 244, 247, 250, 252, 253, 254, 254, 253, 252,
    251, 249, 246, 243, 240, 237, 233, 228, 223, 219, 214, 208, 203, 196, 190, 184,
    178, 171, 165, 158, 151, 144, 138, 129, 120, 112, 103, 95, 87, 79, 71, 64,
    57, 51, 45, 39, 35, 30, 27, 24, 21, 20, 19, 19, 21, 23, 27, 32,
    37, 44, 51, 59, 68, 77, 86, 97, 107, 118, 125, 131, 137, 144, 150, 156,
    162, 168, 174, 180, 185, 191, 197, 202, 207, 212, 217, 222, 227, 231, 235, 238,
    242, 245, 248, 251, 253, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
]
_THERMAL_G = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 5, 9, 13, 17, 22,
    27, 32, 38, 44, 50, 57, 63, 70, 77, 84, 91, 98, 106, 113, 121, 128,
    136, 144, 152, 160, 168, 176, 183, 191, 198, 205, 212, 218, 224, 230, 235, 240,
    245, 249, 253, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 254, 254, 254, 254, 254, 254,
    254, 254, 254, 254, 254, 254, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 254, 254, 252, 252, 250, 249, 248, 246, 245, 243,
    241, 239, 237, 235, 233, 229, 226, 222, 218, 213, 208, 203, 198, 192, 187, 181,
    175, 169, 162, 156, 149, 143, 136, 129, 122, 116, 109, 102, 95, 89, 82, 76,
    70, 63, 57, 51, 45, 40, 35, 29, 25, 20, 16, 12, 8, 5, 2, 2,
]
_THERMAL_B = [
    0, 115, 116, 118, 120, 122, 124, 126, 128, 131, 133, 136, 139, 141, 144, 147,
    151, 154, 157, 160, 164, 167, 170, 174, 177, 181, 184, 188, 194, 200, 206, 211,
    217, 222, 227, 232, 236, 240, 244, 248, 251, 253, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 254, 253, 252,
    251, 250, 248, 247, 246, 245, 243, 242, 241, 240, 239, 238, 237, 236, 235, 235,
    235, 234, 234, 234, 234, 234, 235, 236, 237, 238, 239, 240, 241, 243, 244, 246,
    247, 249, 250, 252, 253, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254,
    254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 252, 249, 246,
    243, 239, 236, 231, 227, 223, 218, 213, 208, 203, 198, 192, 187, 181, 175, 169,
    163, 157, 151, 145, 138, 132, 126, 118, 110, 102, 94, 87, 79, 72, 65, 58,
    51, 45, 38, 32, 27, 22, 17, 13, 8, 5, 2, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 3, 5, 8, 10, 12, 14, 16, 17, 19,
    21, 23, 25, 26, 28, 30, 31, 33, 34, 36, 37, 38, 39, 40, 41, 42,
    43, 43, 43, 44, 44, 44, 44, 43, 43, 42, 42, 41, 40, 40, 39, 38,
    37, 36, 34, 33, 32, 31, 30, 28, 27, 25, 24, 23, 21, 20, 19, 17,
    16, 15, 13, 12, 11, 10, 9, 7, 6, 6, 5, 4, 3, 3, 2, 2,
]
# fmt: on


def thermal_colormap():
    """Return the ImageJ 'Thermal (edited)' LUT as (3, 256) uint16 array."""
    lut = np.array([_THERMAL_R, _THERMAL_G, _THERMAL_B], dtype=np.uint16)
    lut *= 256
    return lut


def build_density_stack(density_dir, genes, output_path, *,
                        sigma=DEFAULT_SIGMA,
                        display_min=DEFAULT_DISPLAY_MIN,
                        display_max=DEFAULT_DISPLAY_MAX,
                        sort=False):
    """Load per-gene density TIFs, blur, and write an ImageJ composite stack.

    Parameters
    ----------
    density_dir : str
        Directory containing ``<gene>.tif`` files.
    genes : list[str]
        Ordered gene names (without ``.tif`` extension).
    output_path : str
        Output TIFF path.
    sigma : float
        Gaussian blur sigma (0 to skip).
    display_min, display_max : float
        ImageJ display range.
    sort : bool
        If True, sort genes alphabetically; otherwise keep input order.
    """
    if sort:
        genes = sorted(genes)
    slices = []
    valid_genes = []

    for gene in genes:
        path = os.path.join(density_dir, f"{gene}.tif")
        if not os.path.isfile(path):
            logger.warning("Density file not found, skipping: %s", path)
            continue
        img = imread(path).astype(np.float32)
        if sigma > 0:
            img = gaussian_filter(img, sigma=sigma)
        img = np.clip(np.round(img), 0, 65535).astype(np.uint16)
        slices.append(img)
        valid_genes.append(gene)

    if not slices:
        raise ValueError(f"No valid density files found in {density_dir}")

    stack = np.stack(slices, axis=0)
    logger.info("Stack shape: %s (%d genes)", stack.shape, len(valid_genes))

    imwrite(
        output_path,
        stack,
        imagej=True,
        photometric='minisblack',
        colormap=thermal_colormap(),
        metadata={
            'axes': 'ZYX',
            'min': display_min,
            'max': display_max,
            'loop': False,
            'Labels': [f'{g}.tif' for g in valid_genes],
            'Properties': {'CurrentLUT': 'Thermal (edited)'},
        },
    )
    logger.info("Written: %s", output_path)


def run_pipeline(run_id, *, density_label=None, threshold=DEFAULT_THRESHOLD,
                 gene_file=None, use_all=False, output=None,
                 sigma=DEFAULT_SIGMA, display_min=DEFAULT_DISPLAY_MIN,
                 display_max=DEFAULT_DISPLAY_MAX, sort=False):
    """CLI entry point."""
    if density_label is None:
        density_label = str(threshold)
    dest_dir = os.path.join(BASE_DEST_DIRECTORY, f'{run_id}_processed')
    read_dir = os.path.join(dest_dir, 'readout')
    density_dir = os.path.join(read_dir, f'density_{density_label}')

    if not os.path.isdir(density_dir):
        raise FileNotFoundError(f"Density directory not found: {density_dir}")

    if use_all:
        genes = sorted(
            Path(f).stem for f in os.listdir(density_dir)
            if f.lower().endswith('.tif')
        )
        file_label = 'all'
    else:
        with open(gene_file, encoding='utf-8') as fh:
            genes = [line.strip() for line in fh if line.strip()]
        file_label = Path(gene_file).stem

    if output is None:
        output = f'density_{density_label}_{file_label}.tif'

    output_path = os.path.join(read_dir, output)

    logger.info("=" * 60)
    logger.info("Density Stack Builder")
    logger.info("=" * 60)
    logger.info("Run ID: %s", run_id)
    logger.info("Density dir: %s", density_dir)
    logger.info("Genes: %d from %s", len(genes), 'directory' if use_all else gene_file)
    logger.info("Output: %s", output_path)
    logger.info("Sigma: %.2f, Display range: %.1f - %.1f", sigma, display_min, display_max)

    build_density_stack(
        density_dir, genes, output_path,
        sigma=sigma, display_min=display_min, display_max=display_max,
        sort=sort,
    )


def main():
    """`python -m sprintseq.cli.density_stack` fallback entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )

    parser = argparse.ArgumentParser(description='Build composite density stack TIFF')
    parser.add_argument('--run-id', type=str, required=True)
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument('--gene-file', type=str, help='Text file with one gene name per line')
    grp.add_argument('--all', dest='use_all', action='store_true', help='Use all density TIFs')
    parser.add_argument('--threshold', type=float, default=None)
    parser.add_argument('-Q', '--quality', type=int, default=None,
                        help='Phred quality score; reads from density_Q<N>/ directory')
    parser.add_argument('--output', type=str, default=None, help='Output filename (written to readout/)')
    parser.add_argument('--sigma', type=float, default=DEFAULT_SIGMA)
    parser.add_argument('--display-min', type=float, default=DEFAULT_DISPLAY_MIN)
    parser.add_argument('--display-max', type=float, default=DEFAULT_DISPLAY_MAX)
    parser.add_argument('--sort', action='store_true', help='Sort genes alphabetically instead of codebook order')
    args = parser.parse_args()

    from sprintseq.cli import resolve_threshold_and_label
    quality = args.quality if args.quality is not None else (
        DEFAULT_QUALITY if args.threshold is None else None)
    _, label = resolve_threshold_and_label(
        args.threshold or DEFAULT_THRESHOLD, quality)
    run_pipeline(
        args.run_id, density_label=label,
        gene_file=args.gene_file, use_all=args.use_all, output=args.output,
        sigma=args.sigma, display_min=args.display_min, display_max=args.display_max,
        sort=args.sort,
    )


if __name__ == "__main__":
    main()
