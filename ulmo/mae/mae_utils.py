""" Utility methods for MAE """

import os

def img_filename(t_per:int, p_per:int,
                 mae_img_path = 's3://llc/mae/PreProc'):
    """Generate the image filename from the
    percentiles

    Args:
        t_per (int):
            Patch percentile of the model used 
        p_per (int):
            Patch percentile of the data
        mae_img_path (str, optional):
            s3 path to the image file

    Returns:
        str: filename with s3 path
    
    """
    base_name = f'mae_reconstruct_t{t_per:02d}_p{p_per:02d}.h5'
    img_file = os.path.join(mae_img_path, base_name)

    return img_file

def parse_mae_img_file(img_file:str):
    """ Grab the train and patch percentages from the image filename

    Args:
        img_file (str): 
            Assumes mae_reconstruct_tXX_pXX.h5 format
    """

    prs = os.path.basename(img_file).parse('_')

    # Train %
    assert prs[2][0] == 't'
    t_per = prs[2][1:]

    # Patch %
    assert prs[3][0] == 'p'
    p_per = prs[3][1:]

    return t_per, p_per
