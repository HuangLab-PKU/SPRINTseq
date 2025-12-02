from skimage.io import imread
from skimage.transform import resize
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({
    "pgf.texsystem": "xelatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'figure.dpi': 300,
})
import PIL
PIL.Image.MAX_IMAGE_PIXELS = 933120000


def crop_df(df,x_start=4000,y_start=4000,x_width=3000,y_width=3000):
    df = df[np.logical_and(
        y_start <= df['Y'].to_numpy(), df['Y'].to_numpy() < y_start+y_width)]
    df = df[np.logical_and(
        x_start <= df['X'].to_numpy(), df['X'].to_numpy() < x_start+x_width)]
    df['Y'] = df['Y'] - y_start
    df['X'] = df['X'] - x_start
    return df

def draw_spots(df,im,savename):
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.imshow(im, cmap='gray')
    sns.scatterplot(data=df, x='X', y='Y', hue='Gene', style='Gene', ax=ax, s=2, edgecolor='none')
    ax.axis('off')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(savename)
    plt.close()

if __name__ == "__main__":
    df = pd.read_csv('genes_unstack_0906.csv')[['Y','X','Gene']]
    df = df.groupby('Gene').sample(frac=0.1)
    df['Y'] = df['Y'] // 10
    df['X'] = df['X'] // 10
    # df = crop_df(df)
    im = imread('cell_map.png')
    image_resized = resize(im, (im.shape[0] // 10, im.shape[1] // 10),
                       anti_aliasing=True)
    draw_spots(df,image_resized,'colored_spots_downsample.pdf')
