import matplotlib.pyplot as plt

def plot_colored_dots(df, save_path):
    """
    Plot the dataframe df with colored dots.
    The dataframe should have the following columns:
    - x: the x coordinate of the point
    - y: the y coordinate of the point
    - class: the class index of the point
    """
    # create a figure and axis
    fig, ax = plt.subplots()
    
    # plot the points with the class as the color
    ax.scatter(df['x'], df['y'], c=df['class'])
    
    # save the plot to the save_path
    plt.savefig(save_path)

    # clear the plot to avoid overlap with future plots
    plt.clf()