import numpy as np

def shuffle_classes(df):
    """
    Shuffle the classes of the dataframe df.
    """
    new_df = df.copy()
    new_df['class'] = np.random.permutation(df['class'])
    return new_df

def bootstrap_classes(df):
    """
    Bootstrap the classes of the dataframe df.
    """
    new_df = df.copy()
    new_df['class'] = np.random.choice(df['class'], len(df))
    return new_df