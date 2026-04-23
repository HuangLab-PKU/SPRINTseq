"""
Sequence generation from intensity data using threshold-based method.

This module is part of the threshold_mapping method and generates sequences
by applying fixed thresholds to intensity values.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm


def get_seq_df(intensity_df, thresholds, cyc_num=10, channels=['cy3', 'cy5']):
    """
    Convert an intensity dataframe into a sequence dataframe by thresholding.

    This function is used by the threshold_mapping method to generate sequences
    from intensity data.

    Parameters
    ----------
    intensity_df : pd.DataFrame
        DataFrame with columns 'Y','X' and 'cyc_{n}_{channel}' for
        each cycle and channel in channels.
    thresholds : dict
        Dictionary mapping channel name to threshold value, e.g., {'cy3': 50, 'cy5': 50}
    cyc_num : int
        Number of sequencing cycles (default: 10)
    channels : list
        List of channel names (default: ['cy3', 'cy5'])

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'Y', 'X', 'Sequence'
    """
    coordinates = intensity_df[['Y', 'X']].to_numpy()

    # Build boolean calls per cycle/channel. Be robust to missing/NaN values
    bool_df = pd.DataFrame({'Y': coordinates[:, 0], 'X': coordinates[:, 1]})
    for cyc in tqdm(range(1, cyc_num + 1), desc='Thresholding'):
        for channel in channels:
            col = f'cyc_{cyc}_{channel}'
            # If intensity column is missing or contains NaN, treat as 0 (below threshold)
            if col not in intensity_df.columns:
                bool_series = pd.Series(False, index=range(len(coordinates)))
            else:
                bool_series = (intensity_df[col].fillna(0) >= thresholds[channel])
            bool_df[col] = bool_series.astype(bool).values

    # Map boolean pairs to bases
    base_bool_map = {(True, True): 'A', (True, False): 'T', (False, True): 'C', (False, False): 'G'}
    base_df = pd.DataFrame({'Y': coordinates[:, 0], 'X': coordinates[:, 1]})
    for cyc in tqdm(range(1, cyc_num + 1), desc='Base calling'):
        c3 = bool_df[f'cyc_{cyc}_cy3']
        c5 = bool_df[f'cyc_{cyc}_cy5']
        # Ensure pairs are Python bool tuples so mapping won't produce NaN
        pairs = [(bool(a), bool(b)) for a, b in zip(c3, c5)]
        base_df[f'cyc_{cyc}'] = pd.Series(pairs).map(base_bool_map).astype(str).values

    seq_df = pd.DataFrame({'Y': coordinates[:, 0], 'X': coordinates[:, 1]})
    seq_df['Sequence'] = base_df[[f'cyc_{i}' for i in range(1, cyc_num + 1)]].agg(''.join, axis=1)
    return seq_df

