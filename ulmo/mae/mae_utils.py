""" Utility methods for MAE """

import os

def img_filename(t_per:int, p_per:int,
                 mae_img_path = 's3://llc/mae/PreProc'):
    base_name = f'mae_reconstruct_t{t_per}_p{p_per}.h5'
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
