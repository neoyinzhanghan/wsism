import pandas as pd
import random
import numpy as np

from synthetic_data.config import SpatialUniformConfig

def get_spacial_uniform_data():
    """
    Return a a dataframe with spatial uniform data.
    The dataframe should have the following columns:
    - x: the x coordinate of the point
    - y: the y coordinate of the point
    - TL_x: the TL_x coordinate of the patch that the point is in (for now None)
    - TL_y: the TL_y coordinate of the patch that the point is in (for now None)
    - class: the class index of the point
    """
    config = SpatialUniformConfig()

    # generate config.num_points uniform random points in the x and y dimensions
    x = np.random.uniform(0, config.x_dim, config.num_points)
    y = np.random.uniform(0, config.y_dim, config.num_points)

    # generate config.num_points random classes
    classes = np.random.choice(range(config.num_classes), config.num_points, p=config.class_probs)

    # create a dataframe with the x, y, and class columns
    df = pd.DataFrame({'x': x, 'y': y, 'class': classes})

    return df
