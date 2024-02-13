from experiments.nn_same_class import run_nn_same_class_experiment
from synthetic_data.spatial_uniform import get_spacial_uniform_data
from synthetic_data.one_class_cluster import get_one_class_cluster

# get the spatial uniform data
df1 = get_spacial_uniform_data()

# run the nearest neighbour experiment with the same class perturbation
run_nn_same_class_experiment(df1, 'spatial_uniform', niters=150)

# get the one class cluster data
df2 = get_one_class_cluster()

# run the nearest neighbour experiment with the same class perturbation
run_nn_same_class_experiment(df2, 'one_class_cluster', niters=150)