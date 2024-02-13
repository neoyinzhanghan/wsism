# make a data class for spatial uniform data
class SpatialUniformConfig:
    def __init__(self):
        self.x_dim = 207360
        self.y_dim = 105984
        self.num_patches = 100
        self.patch_size = 512
        self.num_points = 200 # number of points, typically the number of cells are capped at 3000 for both specimens so we don't need more than 3000 for simulation
        # self.class_probs = [1, 0, 0, 0, 0] # for sanity check
        self.class_probs = [0.5, 0.25, 0.25]
        assert sum(self.class_probs) == 1, "Class probabilities must sum to 1"
        self.num_classes = len(self.class_probs)

class OneClassClusterConfig:
    def __init__(self):
        self.x_dim = 207360
        self.y_dim = 105984
        self.num_patches = 100
        self.patch_size = 512
        self.num_points = 200
        # self.class_probs = [1, 0, 0, 0, 0] # for sanity check
        self.class_probs = [0.5, 0.25, 0.25]
        assert sum(self.class_probs) == 1, "Class probabilities must sum to 1"
        self.num_classes = len(self.class_probs)
        self.cluster_center_x = self.x_dim // 2
        self.cluster_center_y = self.y_dim // 2
        self.cluster_std = 10000