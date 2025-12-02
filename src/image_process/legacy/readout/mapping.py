from reference_check import hamming
from reference_check import read_ref_list
import pandas as pd

def correct_barcode(barcode,pool,exact):
    if barcode in pool:
        return barcode
    else:
        out = []
    for seq in pool:
        if hamming(seq,barcode) <= 1:
            out.append(seq)
    if len(out) > 1:
        return 'Ambiguous'
    else:
        return out[0]

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
    df = df[df['Match']!='Ambiguous']
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
