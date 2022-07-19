#imports
import pandas
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from ulmo.llc import io as llc_io
from ulmo import plotting

def gallery( tbl, a=3, b=3, dLL=20, south=0, north=2, mid_lon=100, dlon=5, tmin=True, tmax=True, title=True): 

    """ Creates a x b gallery of randomly chosen images 
    that fall into med LL +/- dLL (default is +/-20) subset  
    
    south:         decimal number (southern boundary); defaults to north equatorial pacific
    north:         decimal number (southern boundary)
    mid_lon:       decimal number (middle lon value)
    dlon:          decimal number (1/2 of lon range)
    tbl:           pd.DataFrame
    a,b:           decimal number; default is 3x3
    tmin, tmax:    decimal number; defaults to maximum temp range
    title:         string
    
    Returns a plot with all imgs on same color scale, temp scale, median LL"""
    
    #pick a geographical region
    rect = (tbl.lat > south ) & (tbl.lat < north) & (np.abs(tbl.lon + mid_lon) < dlon)
    tbl1 = tbl[ rect ]

    #calculate median LL
    med_LL = np.median(tbl1.LL.to_numpy())
    print('Median LL is {}.'.format(med_LL))

    #restrict dataframe
    
    med = np.abs( tbl1.LL - med_LL) < dLL
    tbl2 = tbl1[ med ]
    
    #pick random dim1 x dim2 imgs
    c = a*b
    list = np.random.choice( tbl2.index.to_numpy(), size = c)
    
    
    #create figure
    
    fig, axes = plt.subplots(a, b, figsize = (8,8) )
    
    if title==True:
        fig.suptitle('9 ~ med LL imgs', fontsize=15)
    else: 
        fig.suptitle(title, fontsize=15)

    cbar_ax = fig.add_axes([0.95, 0.15, 0.02, 0.7])
    cbar_kws={"orientation": "vertical", "shrink":1, "aspect":40, "label": "T - T$_{mean}$"}
    pal, cm = plotting.load_palette()

    #determine tmax and tmin
    imgs = np.empty((64,64,c))
    LLs  = np.empty(c)
    
    for i, ax in enumerate(axes.flat):
        idx = list[ i ] 
        cutout = tbl.iloc[ idx ] 
        img= llc_io.grab_image(cutout)
        imgs[:,:,i] = img
        LLs[i] = cutout.LL
            
    if tmax==True: 
        tmax = np.max(imgs)

    if tmin==True:
        tmin = np.min(imgs)
    print('Temperature scale is {} to {}.'.format(tmin, tmax))
    
    # plot
        
    for i, ax in enumerate(axes.flat):
        img = imgs[:,:,i]
        sns.heatmap(ax=ax, data=img, xticklabels=[], yticklabels=[], cmap=cm,
                cbar=i == 0,
                vmin=tmin, vmax=tmax,
                cbar_ax=None if i else cbar_ax,
                cbar_kws=None if i else cbar_kws)
        ax.set_title('LL = {}'.format(round(LLs[i])))
        ax.figure.axes[-1].yaxis.label.set_size(15)

    return fig.tight_layout(rect=[0, 0, .9, 1])

    #plt.savefig('med_LL_imgs_LC_NP', dpi = 600)
