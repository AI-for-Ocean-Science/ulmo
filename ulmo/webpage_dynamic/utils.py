""" Bokeh portal for Ocean Sciences.  Based on Itamar Reiss' code 
and further modified by Kate Storrey-Fisher"""
import numpy as np

# For the geography figure


from IPython import embed

def mercator_coord(lat, lon):
    """Function to switch from lat/long to mercator coordinates

    Args:
        lat (float or np.ndarray): [description]
        lon (float or np.ndarray): [description]

    Returns:
        tuple: x, y values in mercator
    """
    r_major = 6378137.000
    x = r_major * np.radians(lon)
    scale = x/lon
    y = 180.0/np.pi * np.log(np.tan(np.pi/4.0 + 
        lat * (np.pi/180.0)/2.0)) * scale
    return x, y


def get_region_points(x_min, x_max, y_min, y_max, datasource, IGNORE_TH=-9999):
    """ Get the points within the box

    Args:
        x_min ([type]): [description]
        x_max ([type]): [description]
        y_min ([type]): [description]
        y_max ([type]): [description]
        datasource ([type]): [description]

    Returns:
        [type]: [description]
    """
    xs = np.array(datasource['xs'])
    ys = np.array(datasource['ys'])
    is_in_box = np.logical_and.reduce([xs >= x_min, xs <= x_max, ys >= y_min, ys <= y_max, ys > IGNORE_TH, xs > IGNORE_TH])
    return np.where(is_in_box)[0]


def get_relevant_objects_coords(datasource):
    """ ??

    Args:
        datasource ([type]): [description]

    Returns:
        [type]: [description]
    """
    IGNORE_TH = -999
    xs = np.array(datasource['xs'])
    ys = np.array(datasource['ys'])
    relevant_objects = np.logical_and.reduce([ys > IGNORE_TH, xs > IGNORE_TH])
    return xs[relevant_objects], ys[relevant_objects], relevant_objects


def get_decimated_region_points(x_min, x_max, y_min, y_max, datasource, DECIMATE_NUMBER, 
    id_key:str='names', IGNORE_TH=-9999):
    """ ??

    Args:
        x_min ([type]): [description]
        x_max ([type]): [description]
        y_min ([type]): [description]
        y_max ([type]): [description]
        datasource ([type]): [description]
        DECIMATE_NUMBER ([type]): [description]

    Returns:
        [type]: [description]
    """
    is_in_box_inds = get_region_points(x_min, x_max, y_min, y_max, datasource, IGNORE_TH=IGNORE_TH)
    print('total points before decimation', len(is_in_box_inds))
    if len(is_in_box_inds) < DECIMATE_NUMBER:
        #return is_in_box_inds
        return [datasource[id_key][r] for r in is_in_box_inds]
    random_objects_ = np.random.choice(is_in_box_inds, DECIMATE_NUMBER, replace=False)
    random_objects = [datasource[id_key][r] for r in random_objects_]
    return random_objects


def remove_ticks_and_labels(figure):
    """Simple cleaning method

    Args:
        figure (bokeh.figure): 
    """
    figure.xaxis.major_label_text_font_size = "0pt"
    figure.xaxis.axis_label_text_font_size = "0pt"
    figure.yaxis.major_label_text_font_size = "0pt"

    figure.xaxis.major_tick_line_color = None  # turn off x-axis major ticks
    figure.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks

    figure.yaxis.major_tick_line_color = None  # turn off y-axis major ticks
    figure.yaxis.minor_tick_line_color = None  # turn off y-axis minor tick
