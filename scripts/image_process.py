import os
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from scipy.io import loadmat

from ..src.image_process.utils.io_utils import get_tif_list
from ..src.image_process.fstack import stack_cyc
from ..src.image_process.cidre import cidre_walk
from ..src.image_process.register import register_meta, register_manual
from ..src.image_process.stitch import patch_tiles, template_stitch, stitch_offset, stitch_manual

SRC_DIR = Path('/mnt/data/raw_images')
BASE_DIR = Path('/mnt/data/local_processed_data')
RUN_ID = 'test_data'
src_dir = SRC_DIR / RUN_ID
dest_dir = BASE_DIR / f'{RUN_ID}_processed'

aif_dir = dest_dir / 'focal_stacked'
sdc_dir = dest_dir / 'background_corrected'
rgs_dir = dest_dir / 'registered'
stc_dir = dest_dir / 'stitched'
matdir = dest_dir / 'TileInfo.mat'

def main():
    dest_dir.mkdir(exist_ok=True)
    # TileInfo
    if not matdir.exists():
        try:
            shutil.copy(os.path.join(src_dir, 'TileInfo.mat'), os.path.join(dest_dir, 'TileInfo.mat'))
        except Exception as e: 
            raise Exception(f"An error occurred copying TileInfo.mat: {e}")

    tile_data = loadmat(matdir)
    variable_names = [name for name in tile_data.keys() if not name.startswith('__')]
    TileX, TileY = int(tile_data[variable_names[0]][0][0]), int(tile_data[variable_names[1]][0][0])

    # focal stack
    raw_cyc_list = list(src_dir.glob('cyc_*'))
    for cyc in raw_cyc_list:
        cyc_num = int(cyc.name.split('_')[1])
        stack_cyc(src_dir, aif_dir, cyc_num)

    # background correction
    cidre_walk(aif_dir, sdc_dir)

    # register
    rgs_dir.mkdir(exist_ok=True)
    ref_cyc_rgs = 1
    ref_chn_rgs = 'cy3'
    ref_dir_rgs = sdc_dir / f'cyc_{ref_cyc_rgs}_{ref_chn_rgs}'
    im_names_rgs = get_tif_list(ref_dir_rgs)

    meta_df = register_meta(str(sdc_dir), str(rgs_dir), ['cy3', 'cy5', 'DAPI'], im_names_rgs, ref_cyc_rgs, ref_chn_rgs)
    meta_df.to_csv(rgs_dir / 'integer_offsets.csv')

    # stitch
    stc_dir.mkdir(exist_ok=True)
    ref_cyc_stc = 1
    ref_chn_stc = 'cy3'
    patch_tiles(rgs_dir/f'cyc_{ref_cyc_stc}_{ref_chn_stc}', TileX * TileY)
    template_stitch(rgs_dir/f'cyc_{ref_cyc_stc}_{ref_chn_stc}', stc_dir, TileX, TileY)

    offset_df = pd.read_csv(rgs_dir / 'integer_offsets.csv')
    offset_df = offset_df.set_index('Unnamed: 0')
    offset_df.index.name = None
    stitch_offset(rgs_dir, stc_dir, offset_df)

    # register and stitch cyc_11_DAPI manually
    register_manual(rgs_dir / 'cyc_10_DAPI', sdc_dir / 'cyc_11_DAPI', rgs_dir / 'cyc_11_DAPI')
    stitch_manual(rgs_dir / 'cyc_11_DAPI', stc_dir, offset_df, 10, bleed=30)


if __name__ == "__main__":
    main()
