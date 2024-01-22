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
from tqdm import tqdm
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
from src.network.rnn_RELU import cRNN as cRNN_ReLu
from src.network.rnn_Sigmoid import cRNN as cRNN_Sigmoid
from get_op_dim import get_weights
from get_op_dim import retrieve_op_dimensions
from get_op_dim import setup_environment
from scipy.linalg import subspace_angles
from matplotlib.ticker import MultipleLocator, FixedLocator
from pathlib import Path
import concurrent.futures

def plot_double_lineplot(x_data, y_data_mean, y_data_var, y_data2_mean, y_data2_var, my_xlabel, 
                         my_ylabel, y_max, display_names=None):
    my_fontsize = 15
    # generate standard line plot
    # x_data: [1 , N], data to plot along x-axis
    # y_data: [M , N], data to plot along y-axis
    # my_title, my_xlabel, my_ylabel: (str), axis labels for title, x- and y-axis

    # format y_data
    if len(y_data_mean.shape) > 1:
        if y_data_mean.shape[0] > y_data_mean.shape[1]:
            y_data_mean = y_data_mean.T
        if y_data2_mean.shape[0] > y_data2_mean.shape[1]:
            y_data2_mean = y_data2_mean.T
    else:
        y_data_mean = np.reshape(y_data_mean, [1, np.shape(y_data_mean)[0]])
        y_data2_mean = np.reshape(y_data2_mean, [1, np.shape(y_data2_mean)[0]])

    if len(y_data_var.shape) > 1:
        if y_data_var.shape[0] > y_data_var.shape[1]:
            y_data_var = y_data_var.T
        if y_data2_var.shape[0] > y_data2_var.shape[1]:
            y_data2_var = y_data2_var.T

    elif len(y_data_var.shape) > 1:
        y_data_var = np.reshape(y_data_var, [1, np.shape(y_data_var)[0]])
        y_data2_var = np.reshape(y_data2_var, [1, np.shape(y_data2_var)[0]])    

    # plot
    fig, ax = plt.subplots(1, 1)#figsize=[7, 7]) 
    plt.grid()
    if display_names is None:
        display_name1 = ''
        display_name2 = ''
    else:
        display_name1 = display_names[0]
        display_name2 = display_names[1]
    ax.plot(x_data, y_data_mean[0, :], "r", linewidth=3,
            label=display_name1)
    ax.plot(x_data, y_data2_mean[0, :], "b", linewidth=3,
            label=display_name2)
    ax.fill_between(range(len(y_data_mean[0,:])),
                    y_data_mean[0,:] - y_data_var[0,:],
                    y_data_mean[0,:] + y_data_var[0,:],
                    color='red', alpha=0.3)
    ax.fill_between(range(len(y_data2_mean[0,:])),
                y_data2_mean[0,:] - y_data2_var[0,:],
                y_data2_mean[0,:] + y_data2_var[0,:],
                color='b', alpha=0.3)
    # labels
    ax.tick_params(axis='both', which='major', labelsize=my_fontsize)
    ax.set_xlabel(my_xlabel, fontsize=my_fontsize)
    ax.set_ylabel(my_ylabel, fontsize=my_fontsize)
    if not (display_names is None):
        plt.legend(fontsize=my_fontsize)

    ax.set_ylim(bottom=0, top=1.1 * y_max)

    return fig, ax

def get_global_operative_dimensions(inputfilename=None, ctxt="all", n_units=None, config=None):
    network_type = 'ctxt'
    dim_type = 'columns'
    usl = UtilsSamplingLocs()
    uod = UtilsOpDims()

    skip_l = 1400 - config["training"]["total_seq_length"]
    sampling_loc_props = usl.get_sampling_location_properties(skip_l = skip_l)

    if inputfilename:
        pass
    else:     
        print("used base inputfilename path")
        inputfilename = os.path.join(os.getcwd(), 'local_operative_dimensions', 'localOpDims_'+network_type+'_'+dim_type+'.h5')
    
    [all_local_op_dims, all_fvals] = uod.load_local_op_dims(inputfilename, n_units, sampling_loc_props, network_type='ctxt')

    # combine local operative dimensions to obtain global operative dimensions 
    sampling_locs_to_combine = ctxt #'all' # options for ctxt network: 'ctxt1' 'ctxt2' 'allPosChoice' 'allNegChoice'
    [global_op_dims, singular_values_of_global_op_dims] = uod.get_global_operative_dimensions(sampling_locs_to_combine, sampling_loc_props, all_local_op_dims, all_fvals)

    return global_op_dims, singular_values_of_global_op_dims

def plot_subspace_angle(ctxt1,ctxt2,n_global_dim, label_x, label_y, plot_name, inputfilenames, path_save_dir, tm, models, config):

    angles_list = []
    for i in range(len(inputfilenames)):
        inputfilename = inputfilenames[i]
        tm.model = models[i]
        global_op_dims_one, singular_values_of_global_op_dims_one = get_global_operative_dimensions(inputfilename=inputfilename, ctxt=ctxt1, n_units=tm.hidden_dims, config=config)
        global_op_dims_two, singular_values_of_global_op_dims_two = get_global_operative_dimensions(inputfilename=inputfilename, ctxt=ctxt2, n_units=tm.hidden_dims, config=config)

        angles = np.empty(global_op_dims_one.shape)
        rows, cols = global_op_dims_one.shape[0], global_op_dims_one.shape[1]
        for i in range(rows):
            for j in range(cols):
                angles[i,j] = np.rad2deg(subspace_angles(global_op_dims_one[:,i].reshape(-1,1), global_op_dims_two[:,j].reshape(-1,1)))[0]

        angles_list.append(angles)
    
    angles = np.mean(angles_list, axis=0)

    fig, ax = plt.subplots()
    plt.imshow(angles[:n_global_dim,:n_global_dim], cmap='viridis', interpolation='nearest', vmin=0, vmax=90, extent=[1,n_global_dim,n_global_dim,1])
    plt.xlabel(r'q$_{i}$ ('+ label_x +r'$_{1}$)')
    plt.ylabel(r'q$_{i}$ ('+ label_y +r'$_{2}$)')
    # Add colorbar
    cbar = plt.colorbar(label="angle (deg)", shrink=0.5)
    cbar.set_ticks(ticks=[0, 30, 60, 90])
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_major_locator(MultipleLocator(10))
    plt.gca().invert_yaxis()
    fig.savefig(os.path.join(path_save_dir,plot_name), bbox_inches="tight")

def get_ctxt_ids(the_model):
    ctxt1IDs, ctxt2IDs = np.empty(0), np.empty(0)
    for loader in [the_model.train_dataloader, the_model.test_dataloader, the_model.val_dataloader]:
        for i, (inputs, _) in enumerate(loader):
            contexts = inputs[:,-1,2].detach().numpy()
            ctxt1IDs = np.append(ctxt1IDs, contexts)
            contexts = inputs[:,-1,3].detach().numpy()
            ctxt2IDs = np.append(ctxt2IDs, contexts)

    # every trial is assigned
    assert( (ctxt2IDs == ((ctxt1IDs + 1) % 2)).all() ) 

    ctxt1IDs = np.where(ctxt1IDs == 1)
    ctxt2IDs = np.where(ctxt2IDs == 1)  

    return ctxt1IDs, ctxt2IDs

def get_choice_ids(the_model, w_in, w_hidden, w_out):
    forwardPass, targets = the_model.run_one_forwardPass_all_sets(w_in, w_hidden, w_out, noise_sigma=0)
    directions = forwardPass["m_z_t"][0,-10:,:].mean(axis=0) > 0
    choice1IDs = np.where(directions)
    choice2IDs = np.where(np.invert(directions))
    return choice1IDs, choice2IDs

def process_dimension(dim_nr, global_op_dims, w_in, w_hidden, w_out, first_IDs, second_IDs, n_units, tm):
    n_Wrr_n_modified = remove_dimension_from_weight_matrix(w_hidden.detach().numpy(), global_op_dims[:, dim_nr + 1:n_units + 1], 'columns')
    forwardPass, targets = tm.run_one_forwardPass_all_sets(w_in, n_Wrr_n_modified, w_out, noise_sigma=0)
    mse_1 = get_mse(forwardPass["m_z_t"], targets, first_IDs)
    mse_2 = get_mse(forwardPass["m_z_t"], targets, second_IDs)
    return dim_nr, mse_1, mse_2

def calculate_specific_mses(inputfilename, w_in, w_hidden, w_out, tm, op_dim_ctxt, first_IDs, second_IDs, config):
    # get global operative dim for ctxt 1
    global_op_dims, singular_values_of_global_op_dims = get_global_operative_dimensions(inputfilename, ctxt=op_dim_ctxt, n_units=tm.hidden_dims, config=config) 
    n_op_dims         = tm.hidden_dims
    n_units           = tm.hidden_dims
    mses_1     = np.full([n_op_dims, 1], np.nan)
    mses_2     = np.full([n_op_dims, 1], np.nan)
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(process_dimension, dim_nr, global_op_dims, w_in, w_hidden, w_out, first_IDs, second_IDs, n_units, tm) for dim_nr in range(n_op_dims)]

        with tqdm(total=n_op_dims) as pbar:
            for future in concurrent.futures.as_completed(futures):
                pbar.update(1)  
                result = future.result()
                dim_nr, mse_1, mse_2 = result
                mses_1[dim_nr, 0] = mse_1
                mses_2[dim_nr, 0] = mse_2


    
    return mses_1, mses_2

def calculate_specific_mses_not_par(inputfilename, w_in, w_hidden, w_out, tm, op_dim_ctxt, first_IDs, second_IDs, config):
    # get global operative dim for ctxt 1
    global_op_dims, singular_values_of_global_op_dims = get_global_operative_dimensions(inputfilename, ctxt=op_dim_ctxt, n_units=tm.hidden_dims, config=config) 
    n_op_dims         = tm.hidden_dims
    n_units           = tm.hidden_dims
    mses_1     = np.full([n_op_dims, 1], np.nan)
    mses_2     = np.full([n_op_dims, 1], np.nan)
    with tqdm(total=n_op_dims) as pbar:
        for dim_nr in range(n_op_dims):
            pbar.update(1)  
            # modify W
            n_Wrr_n_modified = remove_dimension_from_weight_matrix(w_hidden.detach().numpy(), global_op_dims[:,dim_nr+1:n_units+1], dim_type)

            # run modified network
            forwardPass, targets = tm.run_one_forwardPass_all_sets(w_in, n_Wrr_n_modified, w_out, noise_sigma=0)
            
            # get performance measures
            mses_1[dim_nr, 0] = get_mse(forwardPass["m_z_t"], targets, first_IDs)
            mses_2[dim_nr, 0] = get_mse(forwardPass["m_z_t"], targets, second_IDs)
    
    return mses_1, mses_2

def cost_different_settings(inputfilename, weightpath, tm, config):

    w_in, w_hidden, w_out = get_weights(path=weightpath)

    ctxt1_ids, ctxt2_ids = get_ctxt_ids(tm)
    choice1IDs, choice2IDs = get_choice_ids(tm, w_in, w_hidden, w_out)

    op_dim = "ctxt1"
    mses_ctxt1_1, mses_ctxt1_2 = calculate_specific_mses(inputfilename, w_in, w_hidden, w_out, tm, op_dim, ctxt1_ids, ctxt2_ids, config)
    op_dim = "ctxt2"
    mses_ctxt2_1, mses_ctxt2_2 = calculate_specific_mses(inputfilename, w_in, w_hidden, w_out, tm, op_dim, ctxt1_ids, ctxt2_ids, config)
    op_dim = "allPosChoice"
    mses_choice1_1, mses_choice1_2 = calculate_specific_mses(inputfilename, w_in, w_hidden, w_out, tm, op_dim, choice1IDs, choice2IDs, config)
    op_dim = "allNegChoice"
    mses_choice2_1, mses_choice2_2 = calculate_specific_mses(inputfilename, w_in, w_hidden, w_out, tm, op_dim, choice1IDs, choice2IDs, config)
    y_max = np.max([mses_ctxt1_1, mses_ctxt1_2, mses_ctxt2_1, mses_ctxt2_2, mses_choice1_1, mses_choice1_2, mses_choice2_1, mses_choice2_2]) 

    return mses_ctxt1_1, mses_ctxt1_2, mses_ctxt2_1, mses_ctxt2_2, mses_choice1_1, mses_choice1_2, mses_choice2_1, mses_choice2_2, y_max

def save_data(mean_ctxt1_1, mean_ctxt1_2, var_ctxt1_1, var_ctxt1_2, mean_ctxt2_1, mean_ctxt2_2, var_ctxt2_1, var_ctxt2_2,
              mean_choice1_1, mean_choice1_2, var_choice1_1, var_choice1_2, mean_choice2_1, mean_choice2_2, var_choice2_1, var_choice2_2, y_max):
    np.save(os.path.join(path_save_dir, 'mean_ctxt1_1.npy'), mean_ctxt1_1)
    np.save(os.path.join(path_save_dir, 'mean_ctxt1_2.npy'), mean_ctxt1_2)
    np.save(os.path.join(path_save_dir, 'var_ctxt1_1.npy'), var_ctxt1_1)
    np.save(os.path.join(path_save_dir, 'var_ctxt1_2.npy'), var_ctxt1_2)
    np.save(os.path.join(path_save_dir, 'mean_ctxt2_1.npy'), mean_ctxt2_1)
    np.save(os.path.join(path_save_dir, 'mean_ctxt2_2.npy'), mean_ctxt2_2)
    np.save(os.path.join(path_save_dir, 'var_ctxt2_1.npy'), var_ctxt2_1)
    np.save(os.path.join(path_save_dir, 'var_ctxt2_2.npy'), var_ctxt2_2)
    np.save(os.path.join(path_save_dir, 'mean_choice1_1.npy'), mean_choice1_1)
    np.save(os.path.join(path_save_dir, 'mean_choice1_2.npy'), mean_choice1_2)
    np.save(os.path.join(path_save_dir, 'var_choice1_1.npy'), var_choice1_1)
    np.save(os.path.join(path_save_dir, 'var_choice1_2.npy'), var_choice1_2)
    np.save(os.path.join(path_save_dir, 'mean_choice2_1.npy'), mean_choice2_1)
    np.save(os.path.join(path_save_dir, 'mean_choice2_2.npy'), mean_choice2_2)
    np.save(os.path.join(path_save_dir, 'var_choice2_1.npy'), var_choice2_1)
    np.save(os.path.join(path_save_dir, 'var_choice2_2.npy'), var_choice2_2)
    np.save(os.path.join(path_save_dir, 'y_max.npy'), y_max)

def plot_mean_var_plot(inputfilenames, model_paths, models, tm, path_save_dir, config):
    ctxt1_1_list, ctxt1_2_list, ctxt2_1_list, ctxt2_2_list, choice1_1_list, choice1_2_list, choice2_1_list, choice2_2_list, y_max_list = [],[],[],[],[],[],[],[],[]
    for i in range(len(models)):
        tm.model = models[i]
        mses_ctxt1_1, mses_ctxt1_2, mses_ctxt2_1, mses_ctxt2_2, mses_choice1_1, mses_choice1_2, mses_choice2_1, mses_choice2_2, y_max = cost_different_settings(inputfilenames[i], model_paths[i], tm, config)
        ctxt1_1_list.append(mses_ctxt1_1)
        ctxt1_2_list.append(mses_ctxt1_2)
        ctxt2_1_list.append(mses_ctxt2_1)
        ctxt2_2_list.append(mses_ctxt2_2) 
        choice1_1_list.append(mses_choice1_1) 
        choice1_2_list.append(mses_choice1_2)  
        choice2_1_list.append(mses_choice2_1) 
        choice2_2_list.append(mses_choice2_2)  
        y_max_list.append(y_max) 

    ctxt1_1_list, ctxt1_2_list = np.stack(ctxt1_1_list, axis=0), np.stack(ctxt1_2_list, axis=0)
    mean_ctxt1_1, mean_ctxt1_2 = np.mean(ctxt1_1_list, axis=0), np.mean(ctxt1_2_list, axis=0)
    var_ctxt1_1, var_ctxt1_2 = np.std(ctxt1_1_list, axis=0), np.std(ctxt1_2_list, axis=0)

    mean_ctxt2_1, mean_ctxt2_2 = np.mean(ctxt2_1_list, axis=0), np.mean(ctxt2_2_list, axis=0)
    ctxt2_1_list, ctxt2_2_list = np.stack(ctxt2_1_list, axis=0), np.stack(ctxt2_2_list, axis=0)
    var_ctxt2_1, var_ctxt2_2 = np.std(ctxt2_1_list, axis=0), np.std(ctxt2_2_list, axis=0)

    mean_choice1_1, mean_choice1_2 = np.mean(choice1_1_list, axis=0), np.mean(choice1_2_list, axis=0)
    choice1_1_list, choice1_2_list = np.stack(choice1_1_list, axis=0), np.stack(choice1_2_list, axis=0)
    var_choice1_1, var_choice1_2 = np.std(choice1_1_list, axis=0), np.std(choice1_2_list, axis=0)

    mean_choice2_1, mean_choice2_2 = np.mean(choice2_1_list, axis=0), np.mean(choice2_2_list, axis=0)
    choice2_1_list, choice2_2_list = np.stack(choice2_1_list, axis=0), np.stack(choice2_2_list, axis=0)
    var_choice2_1, var_choice2_2 = np.std(choice2_1_list, axis=0), np.std(choice2_2_list, axis=0)

    y_max = np.max(y_max_list)

    save_data(mean_ctxt1_1, mean_ctxt1_2, var_ctxt1_1, var_ctxt1_2, mean_ctxt2_1, mean_ctxt2_2, var_ctxt2_1, var_ctxt2_2,
              mean_choice1_1, mean_choice1_2, var_choice1_1, var_choice1_2, mean_choice2_1, mean_choice2_2, var_choice2_1, var_choice2_2, y_max)


    [fig, ax] = plot_double_lineplot(np.arange(tm.hidden_dims), mean_ctxt1_1, var_ctxt1_1, mean_ctxt1_2, var_ctxt1_2, "rank(W$^{OP_{Ctx}}_{k, 1}$)", "cost", y_max, display_names=["context$_{1}$", "context$_{2}$"])
    fig.savefig(os.path.join(path_save_dir,"cost_on_ctx_1.png"), bbox_inches="tight")
    [fig, ax] = plot_double_lineplot(np.arange(tm.hidden_dims), mean_ctxt2_1, var_ctxt2_1, mean_ctxt2_2, var_ctxt2_2, "rank(W$^{OP_{Ctx}}_{k, 2}$)", "cost", y_max, display_names=["context$_{1}$", "context$_{2}$"])
    fig.savefig(os.path.join(path_save_dir, "cost_on_ctx_2.png"), bbox_inches="tight")
    [fig, ax] = plot_double_lineplot(np.arange(tm.hidden_dims), mean_choice1_1, var_choice1_1, mean_choice1_2, var_choice1_2, "rank(W$^{OP_{Cho}}_{k, 1}$)", "cost", y_max, display_names=["choice$_{1}$", "choice$_{2}$"])
    fig.savefig(os.path.join(path_save_dir, "cost_on_cho_1.png"), bbox_inches="tight")
    [fig, ax] = plot_double_lineplot(np.arange(tm.hidden_dims), mean_choice2_1, var_choice2_1, mean_choice2_2, var_choice2_2, "rank(W$^{OP_{Cho}}_{k, 2}$)", "cost", y_max, display_names=["choice$_{1}$", "choice$_{2}$"])
    fig.savefig(os.path.join(path_save_dir,"cost_on_cho_2.png"), bbox_inches="tight")

def skip_empty_dir(subdir):
    do_skip = True
    for attempt in ["one", "two", "three"]:
        if attempt in subdir:
            do_skip = False
    return do_skip

def get_model(config, path):
    params = config["model"]
    if params["activation_func"] == "tanh":
        model = cRNN(
                config["model"],
                input_s=params["in_dim"],
                output_s=params["out_dim"],
                hidden_s=params["hidden_dims"],
                hidden_noise=params["hidden_noise"]
                )
    elif params["activation_func"] == "relu":
        model = cRNN_ReLu(
                    config["model"],
                    input_s=params["in_dim"],
                    output_s=params["out_dim"],
                    hidden_s=params["hidden_dims"],
                    hidden_noise=params["hidden_noise"]
                    )
    elif params["activation_func"] == "sigmoid":
        model = cRNN_Sigmoid(
                    config["model"],
                    input_s=params["in_dim"],
                    output_s=params["out_dim"],
                    hidden_s=params["hidden_dims"],
                    hidden_noise=params["hidden_noise"]
                    )
    model.load_state_dict(torch.load(path))
    model.eval()

    return model


def get_last_folder(path):
    normalized_path = os.path.normpath(path)
    head, tail = os.path.split(normalized_path)
    last_folder = os.path.basename(head) if tail == "" else tail
    
    return last_folder

def plot_with_saved_data(path_save_dir, tm, newymax, newname=None):

    mean_ctxt1_1 = np.load(os.path.join(path_save_dir, "mean_ctxt1_1.npy"))
    var_ctxt1_1 = np.load(os.path.join(path_save_dir, "var_ctxt1_1.npy"))
    mean_ctxt1_2 = np.load(os.path.join(path_save_dir, "mean_ctxt1_2.npy"))
    var_ctxt1_2 = np.load(os.path.join(path_save_dir, "var_ctxt1_2.npy"))

    mean_ctxt2_1 = np.load(os.path.join(path_save_dir, "mean_ctxt2_1.npy"))
    var_ctxt2_1 = np.load(os.path.join(path_save_dir, "var_ctxt2_1.npy"))
    mean_ctxt2_2 = np.load(os.path.join(path_save_dir, "mean_ctxt2_2.npy"))
    var_ctxt2_2 = np.load(os.path.join(path_save_dir, "var_ctxt2_2.npy"))

    mean_choice1_1 = np.load(os.path.join(path_save_dir, "mean_choice1_1.npy"))
    var_choice1_1 = np.load(os.path.join(path_save_dir, "var_choice1_1.npy"))
    mean_choice1_2 = np.load(os.path.join(path_save_dir, "mean_choice1_2.npy"))
    var_choice1_2 = np.load(os.path.join(path_save_dir, "var_choice1_2.npy"))

    mean_choice2_1 = np.load(os.path.join(path_save_dir, "mean_choice2_1.npy"))
    var_choice2_1 = np.load(os.path.join(path_save_dir, "var_choice2_1.npy"))
    mean_choice2_2 = np.load(os.path.join(path_save_dir, "mean_choice2_2.npy"))
    var_choice2_2 = np.load(os.path.join(path_save_dir, "var_choice1_2.npy"))

    y_max  = np.load(os.path.join(path_save_dir, "y_max.npy"))
    y_max = newymax
    if newname == None:
        [fig, ax] = plot_double_lineplot(np.arange(tm.hidden_dims), mean_ctxt1_1, var_ctxt1_1, mean_ctxt1_2, var_ctxt1_2, "rank(W$^{OP_{Ctx}}_{k, 1}$)", "cost", y_max, display_names=["context$_{1}$", "context$_{2}$"])
        fig.savefig(os.path.join(path_save_dir,"cost_on_ctx_1.png"), bbox_inches="tight")
        [fig, ax] = plot_double_lineplot(np.arange(tm.hidden_dims), mean_ctxt2_1, var_ctxt2_1, mean_ctxt2_2, var_ctxt2_2, "rank(W$^{OP_{Ctx}}_{k, 2}$)", "cost", y_max, display_names=["context$_{1}$", "context$_{2}$"])
        fig.savefig(os.path.join(path_save_dir, "cost_on_ctx_2.png"), bbox_inches="tight")
        [fig, ax] = plot_double_lineplot(np.arange(tm.hidden_dims), mean_choice1_1, var_choice1_1, mean_choice1_2, var_choice1_2, "rank(W$^{OP_{Cho}}_{k, 1}$)", "cost", y_max, display_names=["choice$_{1}$", "choice$_{2}$"])
        fig.savefig(os.path.join(path_save_dir, "cost_on_cho_1.png"), bbox_inches="tight")
        [fig, ax] = plot_double_lineplot(np.arange(tm.hidden_dims), mean_choice2_1, var_choice2_1, mean_choice2_2, var_choice2_2, "rank(W$^{OP_{Cho}}_{k, 2}$)", "cost", y_max, display_names=["choice$_{1}$", "choice$_{2}$"])
        fig.savefig(os.path.join(path_save_dir,"cost_on_cho_2.png"), bbox_inches="tight")
    else:
        [fig, ax] = plot_double_lineplot(np.arange(tm.hidden_dims), mean_ctxt1_1, var_ctxt1_1, mean_ctxt1_2, var_ctxt1_2, "rank(W$^{OP_{Ctx}}_{k, 1}$)", "cost", y_max, display_names=["context$_{1}$", "context$_{2}$"])
        fig.savefig(os.path.join(path_save_dir, newname+"_on_ctx_1.png"), bbox_inches="tight")
        [fig, ax] = plot_double_lineplot(np.arange(tm.hidden_dims), mean_ctxt2_1, var_ctxt2_1, mean_ctxt2_2, var_ctxt2_2, "rank(W$^{OP_{Ctx}}_{k, 2}$)", "cost", y_max, display_names=["context$_{1}$", "context$_{2}$"])
        fig.savefig(os.path.join(path_save_dir, newname+"_on_ctx_2.png"), bbox_inches="tight")
        [fig, ax] = plot_double_lineplot(np.arange(tm.hidden_dims), mean_choice1_1, var_choice1_1, mean_choice1_2, var_choice1_2, "rank(W$^{OP_{Cho}}_{k, 1}$)", "cost", y_max, display_names=["choice$_{1}$", "choice$_{2}$"])
        fig.savefig(os.path.join(path_save_dir, newname+"_on_cho_1.png"), bbox_inches="tight")
        [fig, ax] = plot_double_lineplot(np.arange(tm.hidden_dims), mean_choice2_1, var_choice2_1, mean_choice2_2, var_choice2_2, "rank(W$^{OP_{Cho}}_{k, 2}$)", "cost", y_max, display_names=["choice$_{1}$", "choice$_{2}$"])
        fig.savefig(os.path.join(path_save_dir, newname+"_on_cho_2.png"), bbox_inches="tight")

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--config", help="Config file to use", default="rnn.yaml"
    )
    args = parser.parse_args()

    config = load_config(CONFIG_DIR / args.config)
    set_seed(config["experiment"]["seed"])

    directory = os.path.join("..", "..", "io", "output", "rnn1", "trained_models")
    inputfilename1, inputfilename2, inputfilename3 = None, None, None
    model_path1, model_path2, model_path3 = None, None, None
    parent_dir = None
    for subdir, dirs, files in os.walk(directory):
        if skip_empty_dir(subdir):
            continue
        
        if "one" in subdir:
            path_child = Path(subdir)
            parent_dir = path_child.parent.absolute()
            for file in files:
                filepath = subdir + os.sep + file
                if filepath.endswith(".pt"):
                    model_path1 = filepath
                if filepath.endswith(".h5"):
                    inputfilename1 = filepath
 
        if "two" in subdir:
            for file in files:
                filepath = subdir + os.sep + file
                if filepath.endswith(".pt"):
                    model_path2 = filepath
                if filepath.endswith(".h5"):
                    inputfilename2 = filepath

        if "three" in subdir:
            for file in files:
                filepath = subdir + os.sep + file
                if filepath.endswith(".pt"):
                    model_path3 = filepath
                if filepath.endswith(".h5"):
                    inputfilename3 = filepath
        
        if(inputfilename1 == None or inputfilename2 == None or inputfilename3 == None):
            continue
        #print(parent_dir)
        Path(os.path.join(parent_dir, "combined")).mkdir(parents=True, exist_ok=True)

        path_save_dir = os.path.join(parent_dir, "combined")

        # settings (same for all 3)
        _, w_hidden, _ = get_weights(path=model_path1)
        n_global_dim = 30
        if w_hidden.shape[0] < 30:
            n_global_dim = int(w_hidden.shape[0]) 
        config["experiment"]["model"]["hidden_dims"] = w_hidden.shape[0]
        config["experiment"]["training"]["hidden_dims"] = w_hidden.shape[0]
        if "original" in subdir:
            print("changed length for original")
            config["experiment"]["training"]["total_seq_length"] = 1400
            config["experiment"]["options"]["length_skipped_in_data"] = 0

        if "sigmoid" in subdir:
            config["experiment"]["model"]["activation_func"] = "sigmoid"  
        if "relu" in subdir:
            config["experiment"]["model"]["activation_func"] = "relu"  
        network_type = 'ctxt'
        dim_type = 'columns'
    
        model, optimizer = None, None
        tm1, optimizer1 = setup_environment(config["experiment"], model_path1, model, optimizer)
        #allows the model to switch
        model1 = get_model(config["experiment"], model_path1)
        model2 = get_model(config["experiment"], model_path2)
        model3 = get_model(config["experiment"], model_path3)

        models = [model1, model2, model3]
        
        inputfilenames = [inputfilename1, inputfilename2, inputfilename3]
        model_paths = [model_path1, model_path2, model_path3]

        ctxt1 = "allPosChoice"
        ctxt2 = "allNegChoice"
        label_x = "choice"
        label_y = "choice"
        plot_name = "choice.png"
        #plot_subspace_angle(ctxt1,ctxt2,n_global_dim,label_x,label_y,plot_name, inputfilenames, path_save_dir, tm1, models, config["experiment"])
        ctxt1 = "ctxt1"
        ctxt2 = "ctxt2"
        label_x = "ctxt"
        label_y = "ctxt"
        plot_name = "ctxt_s.png"
        #plot_subspace_angle(ctxt1,ctxt2,n_global_dim,label_x,label_y,plot_name, inputfilenames, path_save_dir, tm1, models, config["experiment"])

        #plot_mean_var_plot(inputfilenames, model_paths, models, tm1, path_save_dir, config["experiment"])
        # used to load already made data and set a new ymax/name
        plot_with_saved_data(path_save_dir, tm1, newymax=1.2, newname=get_last_folder(parent_dir))

        inputfilename1, inputfilename2, inputfilename3 = None, None, None
        model_path1, model_path2, model_path3 = None, None, None
        parent_dir = None
        plt.close(fig="all")