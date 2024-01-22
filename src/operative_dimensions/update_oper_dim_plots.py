from samplinglocs import UtilsSamplingLocs
from utilsio import UtilsIO
from utilsplotting import UtilsPlotting
from utilsopdims import UtilsOpDims
from matplotlib import pyplot as plt
from make_unit_length import make_unit_length
from get_neg_deltaFF import get_neg_deltaFF
from sklearn.decomposition import PCA
from remove_dimension_from_weight_matrix import remove_dimension_from_weight_matrix
from get_state_distance_between_trajs import get_state_distance_between_trajs
from get_mse import get_mse
import torch
import argparse
import sys
import os
import numpy as np
sys.path.append('../../')
from src.runner import run
from src.utils import load_config, set_seed
from src.constants import CONFIG_DIR, PROJECT_ROOT
from src.optimizer import optimizer_creator
from src.network.rnn import cRNN
from src.training.training_rnn_ext1 import RNNTrainingModule1
from datetime import datetime
#
from get_op_dim import get_weights
from get_op_dim import retrieve_op_dimensions
from get_op_dim import setup_environment
from scipy.linalg import subspace_angles
from matplotlib.ticker import MultipleLocator, FixedLocator

def skip_empty_dir(subdir):
    do_skip = True
    for attempt in ["one", "two", "three"]:
        if attempt in subdir:
            do_skip = False
    return do_skip

if __name__ == "__main__":    
    directory = os.path.join("..", "..", "io", "output", "rnn1", "done_trained_models")
    for subdir, dirs, files in os.walk(directory):
        if skip_empty_dir(subdir):
            continue
        inputfilename = None
        plot_path = None
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith(".png"):

                if filepath.endswith("cost_on_cho_1.png"):
                    print("found")
                    data1 = PIL.image.open(filepath)
                    print(data1.shape)
                    assert(False)
        path_save_dir = subdir
        #print(path_save_dir, plot_path, inputfilename)

    

