from pathlib import Path
from unittest.mock import patch
import pandas as pd
from lib.utils.io_utils import get_tif_list
from lib.fstack import stack_cyc
from lib.cidre import cidre_walk
from lib.register import register_meta
from lib.stitch import patch_tiles
from lib.stitch import template_stitch
from lib.stitch import stitch_offset

from lib.register import register_manual
from lib.stitch import stitch_manual

from skimage.io import imread
from skimage.io import imsave

SRC_DIR = Path('/mnt/data/raw_images/archive')
BASE_DIR = Path('/mnt/data/local_processed_data')
RUN_ID = '20230810_SPRINT_Qihai_Test_1'
src_dir = SRC_DIR / RUN_ID
dest_dir = BASE_DIR / f'{RUN_ID}_processed'

aif_dir = dest_dir / 'focal_stacked'
sdc_dir = dest_dir / 'background_corrected'
rgs_dir = dest_dir / 'registered'
stc_dir = dest_dir / 'stitched'
rsz_dir = dest_dir / 'resized'

def main():
    raw_cyc_list = list(src_dir.glob('cyc_*'))
    for cyc in raw_cyc_list:
        cyc_num = int(cyc.name.split('_')[1])
        stack_cyc(src_dir, aif_dir, cyc_num)

    cidre_walk(aif_dir, sdc_dir)

    rgs_dir.mkdir(exist_ok=True)
    ref_cyc = 1
    ref_chn = 'cy3'
    ref_dir = sdc_dir / f'cyc_{ref_cyc}_{ref_chn}'
    im_names = get_tif_list(ref_dir)

    meta_df = register_meta(
        str(sdc_dir), str(rgs_dir), ['cy3', 'cy5', 'DAPI'], im_names, ref_cyc, ref_chn)
    meta_df.to_csv(rgs_dir / 'integer_offsets.csv')

    patch_tiles(rgs_dir/f'cyc_{ref_cyc}_{ref_chn}',5*5)
    stc_dir.mkdir(exist_ok=True)
    template_stitch(rgs_dir/f'cyc_{ref_cyc}_{ref_chn}', stc_dir, 5,5)

    offset_df = pd.read_csv(rgs_dir / 'integer_offsets.csv')
    offset_df = offset_df.set_index('Unnamed: 0')
    offset_df.index.name = None
    stitch_offset(rgs_dir, stc_dir, offset_df)

    #register_manual(rgs_dir/'cyc_1_cy3', rsz_dir/'cyc_1_FAM', rgs_dir/'cyc_1_FAM')
    ##stitch_manual(rgs_dir/'cyc_11_DAPI', stc_dir, offset_df, 10, bleed=30)
    ##im = imread(stc_dir/'cyc_11_DAPI.tif')
    ##im_crop = im[10000:20000,10000:20000]
    ##imsave(stc_dir/'cyc_11_DAPI_crop.tif', im_crop)


def test():
    rgs_dir.mkdir(exist_ok=True)
    ref_cyc = 1
    ref_chn = 'cy3'
    ref_dir = sdc_dir / f'cyc_{ref_cyc}_{ref_chn}'
    im_names = get_tif_list(ref_dir)
    #meta_df = register_meta(
    #    sdc_dir, rgs_dir, ['cy3', 'cy5', 'DAPI'], im_names, ref_cyc, ref_chn)
    #meta_df.to_csv(rgs_dir / 'integer_offsets.csv')
    
    stc_dir.mkdir(exist_ok=True)
    patch_tiles(rgs_dir/'cyc_1_cy3',21*15)
    template_stitch(rgs_dir/'cyc_1_cy3', stc_dir, 21, 15)

    offset_df = pd.read_csv(rgs_dir / 'integer_offsets.csv')
    offset_df = offset_df.set_index('Unnamed: 0')
    offset_df.index.name = None
    stitch_offset(rgs_dir, stc_dir, offset_df)


def stitch_test():
    offset_df = pd.read_csv(rgs_dir / 'integer_offsets.csv')
    offset_df = offset_df.set_index('Unnamed: 0')
    offset_df.index.name = None
    register_manual(rgs_dir/'cyc_10_DAPI', sdc_dir/'cyc_11_DAPI', rgs_dir/'cyc_11_DAPI')
    stitch_manual(rgs_dir/'cyc_11_DAPI', stc_dir, offset_df, 10, bleed=30)
    im = imread(stc_dir/'cyc_11_DAPI.tif')
    im_crop = im[10000:20000,10000:20000]
    imsave(stc_dir/'cyc_11_DAPI_crop.tif', im_crop)


if __name__ == "__main__":
    main()
