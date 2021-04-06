########################################################################################
#################### Modules used in the latents exploration ###########################

########################################################################################
### Here we write a funtion 'threshold_get()' to get the left and right threshold
def threshold_get(data_pd, alpha):
    """
    Args:
        data_pd: (pd.dataframe): the sample set
        alpha: (float) the confidence probability

    Returns:
        (left_threshold, right_threshold): (tuple (float, float))
    """
    p_left = (1 - alpha) / 2
    p_right = 1 - (1 - alpha) / 2
    q_left = LL_pd.quantile(p_left)
    q_right = LL_pd.quantile(p_right)
    
    return (q_left, q_right)
########################################################################################

########################################################################################
######## Here we write a function 'hist_tail_create()' to plot the histogram ###########
def hist_tail_create(data_pd, alpha):
    """
    Args:
        data_pd: (pd.dataframe): the sample set
        alpha: (float) the confidence probability
    """
    q_left, q_right = threshold_get(data_pd, alpha)
    sns.histplot(evals_tbl[y2019].LL)
    plt.vlines(x=q_left, ymin=0, ymax=9000, color='red', linestyles='dashed')
    plt.vlines(x=q_right, ymin=0, ymax=9000, color='red', linestyles='dashed')
    plt.xlabel('LL', fontsize=15)
    plt.ylabel('Count', fontsize=15)
    plt.title(f'tail area = 1 - {alpha}', fontsize=25)
########################################################################################

########################################################################################
### Here we write a function 'outliers_preparing()' 
def outliers_preparing(data_pd, alpha):
    """
    Args:
        data_pd: (pd.dataframe): the sample set
        alpha: (float) the confidence probability

    Returns:
        (left_outliers_indices, right_outliers_indices, normal_indices): (tuple (list, list, list))
    """
    q_left, q_right = threshold_get(data_pd, alpha)
    ll_latents_np = data_pd.values.copy()
    left_outliers_indices = np.argwhere(ll_latents_np <= q_left).squeeze()
    right_outliers_indices = np.argwhere(ll_latents_np >= q_right).squeeze()
    normal_indices_left = np.argwhere(q_left <= ll_latents_np).squeeze()
    normal_indices_right = np.argwhere(ll_latents_np <= q_right).squeeze()
    normal_indices = np.intersect1d(normal_indices_left, normal_indices_right)
    
    assert len(left_outliers_indices) + len(right_outliers_indices) + len(normal_indices) == len(ll_latents_np), \
        'the result is not consistent!'
    
    return (left_outliers_indices, right_outliers_indices, normal_indices)
########################################################################################

########################################################################################
### scatter plot create
def scatter_create(latents_embedding, left_outliers_indices, right_outliers_indices, normal_indices):
    """
    Args:
        latents_embedding: (np.array) The array of reduced manifold
        left_outliers_indices: (np.array) The array of left tail indices
        right_outliers_indices: (np.array) The array of right tail indices
        normal_indices: (np.array) The array of normal indices
    """
    plt.scatter(latents_embedding[:, 0][normal_indices], latents_embedding[:, 1][normal_indices],
            s=12, label='normal')
    plt.scatter(latents_embedding[:, 0][left_outliers_indices], latents_embedding[:, 1][left_outliers_indices],
            s=30, label='left outliers')
    plt.scatter(latents_embedding[:, 0][right_outliers_indices], latents_embedding[:, 1][right_outliers_indices],
            s=30, label='right outliers')
    plt.xlabel('1st dimension', fontsize=15)
    plt.ylabel('2ed dimension', fontsize=15)
    plt.legend(fontsize=15, loc=1)
    plt.title('Scatter plot of embedding manifold by UMAP', fontsize=25)
#######################################################################################

#######################################################################################
### Crerate the function 'hist_and_scatter_create()'
def hist_and_scatter_create(latents_embedding, data_pd, alpha):
    """
    Args:
        latents_embedding: (np.array) reduced manifolds
        data_pd: (pd.dataframe): the sample set
        alpha: (float) the confidence probability
    """
    plt.figure(figsize=(22, 8))
    plt.subplot(1, 2, 1)
    hist_tail_create(data_pd, alpha)
    plt.subplot(1, 2, 2)
    left_outliers_indices, right_outliers_indices, normal_indices = outliers_preparing(data_pd, alpha)
    scatter_create(latents_embedding, left_outliers_indices, right_outliers_indices, normal_indices)
#######################################################################################