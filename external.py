#import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D

import seaborn as sns
import os
from tqdm import tqdm
from tqdm import tqdm_notebook
import missingno as msno
from scipy import stats
import gc
import warnings
warnings.filterwarnings("ignore")

#define helping function
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from adjustText import adjust_text

def make_scatter_matrix(in_df):
    sns.set(font_scale = 1)
    #sns.set(style="white")
    def corrfunc(x, y, **kws):
        r, _ = stats.pearsonr(x, y)
        ax = plt.gca()
        ax.annotate("r = {:.3f}".format(r),
                    xy=(.1, .9), xycoords=ax.transAxes)
    
    g = sns.PairGrid(in_df, palette=["red"]) 
    g.map_upper(plt.scatter, s=10)
    g.map_diag(sns.distplot, kde=False) 
    g.map_lower(sns.kdeplot, cmap="Blues_d") 
    g.map_lower(corrfunc)
    plt.show()


def make_pca(in_df, palette, ax, top=500):
    cols = in_df.columns
    pca = PCA(n_components=2)
    
    sorted_mean = in_df.mean(axis=1).sort_values()
    select = sorted_mean.tail(top)
    #print(top)
    in_df = in_df.loc[select.index.values]
    pca.fit(in_df)
    temp_df = pd.DataFrame()
    temp_df['pc_1']=pca.components_[0]
    temp_df['pc_2']=pca.components_[1]
    temp_df.index = cols
    print(pca.explained_variance_ratio_)
    temp_df['color']=palette
    #fig,ax=plt.subplots(figsize=(12,6))
    temp_df.plot(kind='scatter',x='pc_1',y='pc_2',s=30, c=temp_df['color'], ax=ax)
    #print(temp_df.index.values)
       
    texts = [ax.text(temp_df.iloc[i]['pc_1'], 
                       temp_df.iloc[i]['pc_2'],
                       cols[i])
                       for i in range(temp_df.shape[0])]
    
    
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'),ax=ax)
    ax.set_title('PCA', size=14)
    ax.set_xlabel('PC1_{:.3f}'.format(pca.explained_variance_ratio_[0]),size=12)
    ax.set_ylabel('PC2_{:.3f}'.format(pca.explained_variance_ratio_[1]),size=12)
    ax.yaxis.label.set_size(12)
    ax.xaxis.label.set_size(12)
    

    
def make_mds(in_df, palette, ax, top=500):
    cols = in_df.columns
    pca = MDS(n_components=2,metric=True)
    
    sorted_mean = in_df.mean(axis=1).sort_values()
    select = sorted_mean.tail(top)
    #print(top)
    in_df = in_df.loc[select.index.values]
    temp_df = pd.DataFrame(pca.fit_transform(in_df.T),
                                 index=cols,columns =['pc_1','pc_2'] )
    
    temp_df['color']=palette
    
    temp_df.plot(kind='scatter',x='pc_1',y='pc_2',s=50, c=temp_df['color'], ax=ax)
    #print(temp_df.index.values)
       
    texts = [ax.text(temp_df.iloc[i]['pc_1'], 
                       temp_df.iloc[i]['pc_2'],
                       cols[i])
                       for i in range(temp_df.shape[0])]
    
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'),ax=ax)
    ax.set_title('MDS',size=14)
    ax.set_xlabel('DIM_1',size=12)
    ax.set_ylabel('DIM_2',size=12)
    ax.yaxis.label.set_size(12)
    ax.xaxis.label.set_size(12)

    #plt.show()
    
def make_vulcano(df, ax, x='-Log10PValue', y='Log2FC',
                 annot_index=pd.Series(), 
                 annot_names=pd.Series(),title='Vulcano'):
    
    df.plot(kind='scatter', x=x, y=y, ax=ax, alpha=0.2)

    df[(df['PValue']<0.05) & (df['Log2FC']>1)].plot(
    kind='scatter',x=x,y=y, ax=ax, 
        c='g', label='Up in Control',alpha=0.5)

    df[(df['PValue']<0.05) & (df['Log2FC']<-1)].plot(
    kind='scatter', x=x,y=y,
        ax=ax, c='r', label='Up in Senescent',alpha=0.5)
    
    ax.legend()
    ax.set_title(title)
    ax.yaxis.label.set_size(12)
    ax.xaxis.label.set_size(12)
    texts = [ax.text(df.loc[i][x], df.loc[i][y],name)
                       for i,name in zip(annot_index,annot_names)]

    adjust_text(texts, arrowprops=dict(arrowstyle='->',
                                       color='red'),
                ax=ax)

def hist_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    new_handles = [Line2D([], [], c=h.get_edgecolor()) for h in handles]
    ax.legend(handles=new_handles, labels=labels)
   
if __name__ == '__main__':
    pass
    