import os
from pathlib import Path
import pandas as pd



def imlist_to_df(imlist):
    """Returns a organized dataframe of metadata

    Parameters
    ----------
    imlist : list
        List of image names, following a convention of:
        '<cycle>-<tile>-<channel>-<depth>.tif'
        e.g. 'C003-T0015-cy5-Z007.tif'

    Returns
    -------
    df : pandas.DataFrame
        Dataframe of image metadata, with columns:
        Cycle, Tile, Channel, Z, Name

    """
    imlist_2d = [s.strip('.tif').split('-') for s in imlist]
    df = pd.DataFrame(imlist_2d, columns=['Cycle', 'Tile', 'Channel', 'Z'])
    for column in ['Cycle','Tile','Z']:
        df[column] = df[column].apply(lambda x: int(x[1:]))
    df['Name'] = imlist
    return df


def get_tif_list(path, abspath=False):
    """Returns a list of all tif files in a directory

    Parameters
    ----------
    path : str
        Path to directory
    abspath : bool
        If True, return absolute paths

    Returns
    -------
    tif_list : list
        List of tif files

    """
    p = Path(path)
    if abspath:
        tif_list = [str(f) for f in p.glob('*.tif')]
    else:
        tif_list = [f.name for f in p.glob('*.tif')]
    return tif_list


def read_df(path):
    """Returns the dataframe with index column.

    Parameters
    ----------
    path : str
        Path to csv file, having no index column name.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with index.

    """
    df = pd.read_csv(path)
    df.set_index('Unnamed: 0', inplace=True)
    df.index.name = None
    return df


def main():
    p = '/Users/Leon/Downloads/fstack_revisit/cyc_3'
    tif_list = get_tif_list(p)
    df = imlist_to_df(tif_list)
    df_g = df.groupby(['Cycle','Tile','Channel'])['Name']
    for (cyc,tile,chn),group in df_g:
        print(cyc,tile,chn)
        print(list(group))


if __name__ == '__main__':
    main()