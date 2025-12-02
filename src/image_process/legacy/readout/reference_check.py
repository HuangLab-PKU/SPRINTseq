import pandas as pd

def hamming(x,y):
    hamm = 0
    for a,b in zip(x,y):
        if a!=b:
            hamm += 1
    return hamm

def find_match(s,l):
    for seq in l:
        h = hamming(s,seq)
        if h <= 1:
            return True
    else:
        return False

def reference_check(df,l):
    df_less = df.groupby('Sequence').filter(lambda x: find_match(x.name,l))
    return df_less

def filter_by_count(df,a=5,c=9,g=8,t=8):
    for char,cnt in zip(['A','C','G','T'],[a,c,g,t]):
        df = df[df['Sequence'].apply(lambda x: x.count(char)<=cnt)]
    return df

def read_ref_list(filename,require_dict=True):
    df = pd.read_csv(filename)[['Barcode','Gene']]
    if require_dict:
        return dict(zip(df['Barcode'],df['Gene']))
    else:
        return list(df['Barcode'])

def check_sequence(seq_file,ref_file):
    ref = pd.read_csv(ref_file)
    barcodes = list(ref['Barcode'])
    df = pd.read_csv(seq_file)
    df = filter_by_count(df)
    print(f'Reads filtered by count: {len(df)}.')
    df = reference_check(df,barcodes)
    print(f'Reads reference checked: {len(df)}.')
    return df

def main():
    ref = pd.read_csv('plex_map_filtered_0930.csv')
    barcodes = list(ref['Barcode'])
    df = pd.read_csv('raw_sequence.csv')
    print('Raw reads:',len(df))
    df = filter_by_count(df)
    print('Reads filtered by count:',len(df))
    df = reference_check(df,barcodes)
    print('Reference checked:',len(df))
    df.to_csv('ref_checked_0930.csv',index=False)
    

if __name__ == "__main__":
    main()
