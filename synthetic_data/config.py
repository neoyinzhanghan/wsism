# make a data class for spatial uniform data
class SpatialUniformConfig:
    def __init__(self):
        self.x_dim = 100000
        self.y_dim = 100000
        self.num_patches = 100
        self.patch_size = 512
        self.num_points = 3000 # number of points, typically the number of cells are capped at 3000 for both specimens so we don't need more than 3000 for simulation
        self.class_probs = [0.7, 0.1, 0.1, 0.05, 0.05]
        assert sum(self.class_probs) == 1, "Class probabilities must sum to 1"
        self.num_classes = len(self.class_probs)