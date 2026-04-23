from .reference_check import hamming
from .reference_check import read_ref_list
import pandas as pd

def correct_barcode(barcode, pool, exact):
    """
    Correct a barcode by matching it against a reference pool.
    
    Parameters
    ----------
    barcode : str
        Query barcode sequence.
    pool : list or iterable of str
        Reference barcode sequences.
    exact : bool
        If True, require exact match only.
        If False, allow Hamming distance <= 1.
    
    Returns
    -------
    str or None
        - Exact or uniquely corrected barcode string if found
        - 'Ambiguous' if multiple candidates within allowed distance
        - None if no candidate found
    """
    # Exact mode: only accept perfect matches, no fuzzy correction
    if exact:
        if barcode in pool:
            return barcode
        else:
            return None

    # Fuzzy mode: allow Hamming distance <= 1
    out = []
    for seq in pool:
        if hamming(seq, barcode) <= 1:
            out.append(seq)

    if len(out) > 1:
        # Multiple possible corrections -> ambiguous, better to drop
        return 'Ambiguous'
    elif len(out) == 1:
        # Single candidate -> use it
        return out[0]
    else:
        # No candidate -> treat as unmapped
        return None  # No match found

def deplex(g):
    if '+' in g:
        return g.split('+')
    else:
        return [g]

def map_barcode(filename,ref_filename,exact=False):
    df = pd.read_csv(filename)
    ref_list = read_ref_list(ref_filename)
    correct_dict = {x:correct_barcode(x,ref_list,exact=exact) for x in set(df['Sequence'])}
    df['Match'] = df.loc[:,'Sequence'].map(correct_dict)
    # Filter out 'Ambiguous' and None (no match) entries
    df = df[(df['Match'] != 'Ambiguous') & (df['Match'].notna())]
    df['Gene'] = df.loc[:,'Match'].map(read_ref_list(ref_filename,require_dict=True))
    df['Gene'] = df['Gene'].apply(deplex)
    return df

def unstack_plex(df):
    df = df[['Y','X','Gene']]
    df = pd.DataFrame([(tup.Y,tup.X,d) for tup in df.itertuples() for d in tup.Gene])
    df.columns = ['Y','X','Gene']
    return df

if __name__ == "__main__":
    df = map_barcode('ref_checked_1020.csv','plex_map_filtered_1005.csv')
    df = unstack_plex(df)
    df.to_csv('mapped_unstack_1020.csv',index=False)
