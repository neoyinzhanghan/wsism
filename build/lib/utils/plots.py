import os
import matplotlib.pyplot as plt
import seaborn as sns

def make_line_dist_plot(data_list, line, save_dir, title):
    """
    Plot an empirical gaussian kernel density estimation of the distribution of the values in data_list.
    Add a vertical line at the position of line. Save the plot to the save_dir.
    """
    # replace all spaces with underscores in the title and add .png to the end
    plot_save_name = title.replace(" ", "_") + ".png"

    plot_save_path = os.path.join(save_dir, plot_save_name)

    # plot the distribution of the data_list using KDE
    sns.kdeplot(data_list, fill=True, color='g', alpha=0.6)

    # plot a vertical line at the position of line
    plt.axvline(x=line, color='r', linestyle='--')

    # add a title to the plot
    plt.title(title)

    # save the plot to the save_dir
    plt.savefig(plot_save_path)

    # clear the plot to avoid overlap with future plots
    plt.clf()