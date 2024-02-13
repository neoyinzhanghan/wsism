import os
import time

from perturbations.class_perturbations import bootstrap_classes
from utils.plots import make_line_dist_plot
from utils.visualize import plot_colored_dots
from utils.record import save_config
from tqdm import tqdm
from synthetic_data import config

def euclidean_distance(x1, y1, x2, y2):
    """
    Return the euclidean distance between (x1, y1) and (x2, y2).
    """
    return ((x1 - x2)**2 + (y1 - y2)**2)**0.5

def _closest_row(df, row_idx):
    """
    Find the row in df that is the closest in euclidean distance to the row at row_idx.
    Return the index of the closest row.
    """

    # get the x and y coordinates of the row at row_idx
    x1 = df.at[row_idx, 'x']
    y1 = df.at[row_idx, 'y']

    # compute the euclidean distance between the row at row_idx and all other rows
    distances = [euclidean_distance(x1, y1, df.at[i, 'x'], df.at[i, 'y']) for i in range(len(df))]

    # find the index of the row with the smallest distance
    closest_idx = distances.index(min(distances))

    assert closest_idx == row_idx, "The closest row should be the same as the row at row_idx"
    assert min(distances) == 0, "The closest row should be the same as the row at row_idx"

    # find the index of the row with the second smallest distance by setting the distance of the closest row to infinity
    distances[closest_idx] = float('inf')
    second_closest_idx = distances.index(min(distances))

    return second_closest_idx

def _same_class_as_nn(df, row_idx):
    """
    Find the row in df that is the closest in euclidean distance to the row at row_idx.
    Return true iff the class of the closest row is the same as the class of the row at row_idx.
    """

    # get the index of the closest row
    closest_idx = _closest_row(df, row_idx)

    # return true iff the class of the closest row is the same as the class of the row at row_idx
    return df.at[closest_idx, 'class'] == df.at[row_idx, 'class']

def proportion_same_class_as_nn(df):
    """
    Return the proportion of rows in df for which the closest row in euclidean distance has the same class.
    """
    return sum([_same_class_as_nn(df, i) for i in range(len(df))]) / len(df)

def run_nn_same_class_experiment(df, data_set_name, niters=100, base_dir='.'):
    """
    Run the neural network experiment with the same class perturbation.
    """
    proportions = []

    start_time = time.time()

    for i in tqdm(range(niters), desc="Running NN Same Class Proportion Experiment"):
        perturbed_df = bootstrap_classes(df)
        proportions.append(proportion_same_class_as_nn(perturbed_df))

    # calculate the observed proportion
    observed_proportion = proportion_same_class_as_nn(df)

    # save a line plot of the distribution of proportions

    # save_dir should be the location of this package and then /logs
    logs_dir = os.path.join(base_dir, "logs")

    # the log name should be nn_same_class_prop followed by the dataset name surrounded by double under score and then current date and time
    log_name = "nn_same_class_prop" + "__" + data_set_name + "__" + time.strftime("%Y-%m-%d__%H-%M-%S")

    # the save directory should be the logs_dir followed by the log_name
    save_dir = os.path.join(logs_dir, log_name)
    os.makedirs(save_dir, exist_ok=True)
    print("Save Directory: ", save_dir)
    make_line_dist_plot(proportions, observed_proportion, save_dir, "NN Same Class Proportion")

    # save the df to a csv file in the save_dir using data_set_name as the file name and .csv as the file extension
    df.to_csv(os.path.join(save_dir, data_set_name + ".csv"))

    # save a plot of the points in the df to the save_dir using data_set_name as the file name and .png as the file extension
    pic_save_path = os.path.join(save_dir, data_set_name + ".png")
    plot_colored_dots(df, pic_save_path)

    print("Observed Proportion: ", observed_proportion)
    print("Mean Proportion: ", sum(proportions) / len(proportions))
    p_value = sum([prop >= observed_proportion for prop in proportions]) / len(proportions)
    print("P-value: ", p_value)

    time_elapsed = time.time() - start_time

    print("Total runtime: ", time_elapsed)

    # save the observed proportion, mean proportion, p-value, and time elapsed to a file in the save_dir using result as the file name and .txt as the file extension
    with open(os.path.join(save_dir, "result.txt"), "w") as f:
        f.write("Observed Proportion: " + str(observed_proportion) + "\n")
        f.write("Mean Proportion: " + str(sum(proportions) / len(proportions)) + "\n")
        f.write("P-value: " + str(p_value) + "\n")
        f.write("Total runtime: " + str(time_elapsed) + "\n")

        # save the config to the save_dir using config as the file name and .txt as the file extension
        save_config(config, os.path.join(save_dir, "config.yaml"))