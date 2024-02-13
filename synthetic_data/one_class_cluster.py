import pandas as pd
import random
import numpy as np

from synthetic_data.config import OneClassClusterConfig

def get_one_class_cluster():
    """
    Return a a dataframe with spatial uniform data if the class is not 0.
    Ifthe class is 0, return a dataframe with a gaussian cluster of points centered at config.cluster_center_x, config.cluster_center_y with standard deviation config.cluster_std.
    The dataframe should have the following columns:
    - x: the x coordinate of the point
    - y: the y coordinate of the point
    - TL_x: the TL_x coordinate of the patch that the point is in (for now None)
    - TL_y: the TL_y coordinate of the patch that the point is in (for now None)
    - class: the class index of the point
    """
    config = OneClassClusterConfig()

    # generate config.num_points uniform random points in the x and y dimensions
    x = np.ones(config.num_points) * -1
    y = np.ones(config.num_points) * -1

    # generate config.num_points random classes
    classes = np.random.choice(range(config.num_classes), config.num_points, p=config.class_probs)

    # create a dataframe with the x, y, and class columns
    df = pd.DataFrame({'x': x, 'y': y, 'class': classes})

    # traverse the dataframe and replace the x and y coordinates of the points with class 0 with points from a gaussian cluster
    for i in range(len(df)):
        if df.at[i, 'class'] == 0:
            df.at[i, 'x'] = np.random.normal(config.cluster_center_x, config.cluster_std)
            df.at[i, 'y'] = np.random.normal(config.cluster_center_y, config.cluster_std)
        else:
            df.at[i, 'x'] = np.random.uniform(0, config.x_dim)
            df.at[i, 'y'] = np.random.uniform(0, config.y_dim)

    # assert that the x and y coordinates of the points with class 0 have been replaced which means they are all non-negative
    assert all(df[df['class'] == 0]['x'] >= 0), "The x coordinates of the points with class 0 should be non-negative"
    
    return df