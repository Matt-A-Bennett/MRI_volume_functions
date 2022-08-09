#!/usr/bin/env python3

import numpy as np
import scipy.ndimage.filters as f
import nibabel as nib
from itertools import islice
from copy import deepcopy as dc
import operator
import matplotlib.pyplot as plt

def get_truth(input1, comparison_op, input2):
    """
    Compare the values of two objects and return the boolean result

    Inputs:
        input1:         Object which can be compared (using '>', '<', '>=',
                        '<=', '==', '!=') to <input2>

        comparison_op:  <String> ('>', '<', '>=', '<=', '==', '!=') for which
                        comparison to make

        input2:         Object which can be compared (using '>', '<', '>=',
                        '<=', '==', '!=') to <input1>

    Returns:            <Boolean> object
    """

    ops = {'>': operator.gt,
           '<': operator.lt,
           '>=': operator.ge,
           '<=': operator.le,
           '==': operator.eq,
           '!=': operator.ne
          }

    return ops[comparison_op](input1, input2)


def our_func(data, thresh=('>=', -float('inf'))):
    """
    Function to be applied to data. MUST return a single value!

    Inputs:
        data:       1D numpy array

        thresh:     <Tuple> containing two values: <string>  for which
                    comparison to make ('>', '<', '>=', '<=', '==', '!=') and a
                    number to act as the threshold itself (default any number)

    Returns:
        result:     result of the function (if data is empty, returns nan
                    value)
    """

    result = np.nan
    thresh_vals = data[get_truth(data, thresh[0], thresh[1])]
    if thresh_vals.size:
        result = np.mean(data[get_truth(data, thresh[0], thresh[1])])
    return result


def label_do(data, labels, func=np.mean, *args, **kwargs):
    """
    Apply any function (default np.mean) to each set of labelled data voxels
    and return the results as a 1D numpy array

    Inputs:
        data:           Numpy array

        labels:         Numpy array of the same size as data, containing numpy
                        array of non-zero int label value for each voxel

        func:           Function to be applied to each set of voxels labelled
                        in data. The function MUST return a single value! (the
                        default fuction is np.mean)

        *args:          Any non-named arguments to pass to func

        **kwargs:       Any named arguments to pass to func

    Returns:
        func_results:   1D numpy array containing the value returned from the
                        function from each set of labelled voxels
    """

    func_results = np.zeros((len(np.unique(labels))-1))
    for idx, label in enumerate(np.unique(labels[labels>0])):
        func_results[idx] = func(data[labels==label], *args, **kwargs)
    return func_results


def split_by_label(data, labels):
    """
    Put <data> into a dictionary with each label number from <labels> as a
    'key' and a 1D numpy array containing the set of labelled data as the
    'value'

    Inputs:
        data:               Numpy array

        labels:             Numpy array of the same size as data, containing
                            numpy array of non-zero int label value for each
                            voxel

    Returns:
        labelled_voxels:    Dictionary for each set of labelled voxels, with
                            each label number as a 'key' and the 1D numpy
                            array containing the set of labelled data as the
                            'value'
    """

    labelled_voxels = {}
    for label in np.unique(labels[labels>0]):
        labelled_voxels[label] = data[labels==label]
    return labelled_voxels


def chunks(data, n_chunks=10):
    """
    Inputs:
        data:       Dictionary with keys being numbers and vaules being 1D
                    numpy arrays

        n_chunks:   Number of keys for resulting dictionary (default 10)

    Returns:
                    A dictionary containing one key per chunk, and containing
                    all the values from the original keys that fit into each
                    chunk
    """

    size = int(np.ceil((max(data) - min(data))/n_chunks))
    it = iter(data)
    for plot_count in range(0, len(data), size):
        yield {k:data[k] for k in islice(it, size)}


def chunk_summarise(chunk, thresh=('>=', -float('inf'))):
    """
    Takes in a dictionary and calculates the mean and standard deviation of the
    values associated with each key

    Inputs:
        chunk:          Dictionary containing keys as numbers and values as 1D
                        numpy arrays

        thresh:     <Tuple> containing two values: <string>  for which
                    comparison to make ('>', '<', '>=', '<=', '==', '!=') and a
                    number to act as the threshold itself (default any number)

    Returns:
        overall_mean:   Mean of the data passing the threshold

        overall_std:    Standard deviation of the data passing the threshold
    """

    overall_mean = np.nan
    overall_std = np.nan
    overall_vals = np.zeros(1,)
    for values in chunk.values():
        thresh_vals = values[get_truth(values, thresh[0], thresh[1])]
        if thresh_vals.size:
            overall_mean =+ np.mean(thresh_vals)
            overall_vals = np.concatenate((overall_vals, thresh_vals),
                                          axis=None)
    if overall_vals.size > 1:
        overall_std = np.std(overall_vals[1:])
    return overall_mean, overall_std


def binarise_mask(mask, thresh=('>', 0)):
    """
    Take a numpy array and return a boolean mask showing where values are above
    some threshold

    Inputs:
        mask:       Numpy array of zero and non-zero values

        thresh:     <Tuple> containing two values: <string>  for which
                    comparison to make ('>', '<', '>=', '<=', '==', '!=') and a
                    number to act as the threshold itself (default any number
                    greater than zero)

    Returns:
                    Numpy array of boolean values
    """

    return get_truth(mask, thresh[0], thresh[1])


def merge_masks(masks, thresh=('>', 0)):
    """
    Take a list of numpy arrays and sum them to make a single numpy array. Any
    vaules above some threshold (default zero) are set to True and False
    otherwise

    Inputs:
        masks:      <List> of numpy arrays all the same shape

        thresh:     <Tuple> containing two values: <string>  for which
                    comparison to make ('>', '<', '>=', '<=', '==', '!=') and a
                    number to act as the threshold itself (default any number
                    greater than zero)

    Returns:
                    A single numpy array of boolean values
    """

    overall_mask = dc(masks[0])
    for mask in masks[1:]:
        overall_mask += mask
    return binarise_mask(overall_mask, thresh=thresh)


def mask_data(data, mask, non_mask_value=0, thresh=('>', 0)):
    """
    Set any values in <data> falling outside <mask> to <non_mask_value>

    Inputs:
        data:           Numpy array of numbers

        mask:           Numpy array of booleans (or zero and non-zero values)
                        the same shape as data

        non_mask_value: What to set the data values falling outside the mask to
                        (default zero)

        thresh:         <Tuple> containing two values: <string>  for which
                        comparison to make ('>', '<', '>=', '<=', '==', '!=')
                        and a number to act as the threshold itself (default
                        any number greater than zero)

    Returns:
        data:           The original data, but with any values falling outside
                        the mask set to <non_mask_value>
    """

    data, mask = dc(data), dc(mask)
    mask = binarise_mask(mask, thresh=thresh)
    data[~mask] = non_mask_value
    return data


def get_coords(data, thresh=('>=', -float('inf'))):
    """
    Get the XYZ coordinates of all voxels passing thresh into a 2D numpy array
    (coords x voxels)

    Inputs:
        data:       Numpy array

        thresh:     <Tuple> containing two values: <string>  for which
                    comparison to make ('>', '<', '>=', '<=', '==', '!=') and a
                    number to act as the threshold itself (default any number)

    Returns:
        coords:     2D numpy array (coords x voxels)
    """

    thresh_boolean = binarise_mask(data, thresh=thresh)
    coords = np.where(thresh_boolean)
    coords = np.vstack([coords[0], coords[1], coords[2]])
    return coords


def plot_3d(data, thresh=('>=', -float('inf')), lims=None):
    """
    Plot each voxel at its x, y, z coordinate

    Inputs:
        data:       Numpy array of numbers

        thresh:     <Tuple> containing two values: <string>  for which
                    comparison to make ('>', '<', '>=', '<=', '==', '!=')
                    and a number to act as the threshold itself (default
                    any number)

        lims:       <Tuple> containing <tuple> of XYZ lower lim and <tuple> of
                    XYZ upper lims

    Returns:
        fig, ax:    Figure containing the plot
    """

    # get XYZ coordinates of the voxels of interest
    orig_coords = get_coords(data, thresh=thresh)

    # get the values from the masked region
    thresh_boolean = binarise_mask(data, thresh=thresh)
    values = data[thresh_boolean]

    # plot each voxel at its x, y, z coordinate
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.scatter3D(orig_coords[0], orig_coords[1], orig_coords[2],
                 c=values, cmap='hot', s=2)

    if lims:
        ax.set_xlim(lims[0][0], lims[1][0])
        ax.set_ylim(lims[0][1], lims[1][1])
        ax.set_zlim(lims[0][2], lims[1][2])

    return fig, ax


def mask_pca_coords(mask):
    """
    Returns:
        - the x,y,z coordinates (zero-centered) of mask along the pca axes
        - the 3 singular vectors (pointing along the pca components)
        - the original coordinates (zero-centered) in the standard axes system
    """

    # get XYZ coordinates of the voxels of interest
    orig_coords = get_coords(mask, thresh=thresh)

    # zero centre the coords
    mean = orig_coords.mean(axis=1)
    orig_coords = orig_coords - mean[:, np.newaxis]

    # find singular vectors (i.e principal components)
    U, _, _ = np.linalg.svd(orig_coords)

    # encode the original coordinates in the new coordinate system
    new_coords = np.transpose(U) @ orig_coords

    return new_coords, U, orig_coords


def mask_pos_anterior_position(new_coords):
    """
    For each voxel, calculate a number between zero and one according to its
    position along the posterior/anterior axis (returned as a numpy array).
    """

    # the first vector coord tells us distance along posterior/anterior axis
    pos_ant_loading = new_coords[0,:]

    # normalise the distance to between 0 and 1
    pos_ant_loading -= min(pos_ant_loading)
    pos_ant_loading /= max(pos_ant_loading)

    return pos_ant_loading


def plot_pos_anterior_voxels(pos_ant_loading, orig_coords, pos_ant_dir):
    """
    Plot the voxels in their original coordinates. Show the anterior/posterior
    axis (pink line) and colour code each voxel according to it's position
    along this axis.
    """

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    # make everything fit nicely
    axis_lims = np.max(np.abs(orig_coords))*1.1

    # plot the standard axes
    ax.plot3D([-axis_lims, axis_lims], [0, 0], [0, 0], 'black')
    ax.plot3D([0, 0], [-axis_lims, axis_lims], [0, 0], 'black')
    ax.plot3D([0, 0], [0, 0], [-axis_lims, axis_lims], 'black')

    # plot the anterior/posterior direction
    ax.plot3D([pos_ant_dir[0]*-axis_lims, pos_ant_dir[0]*axis_lims],
             [pos_ant_dir[1]*-axis_lims, pos_ant_dir[1]*axis_lims],
             [pos_ant_dir[2]*-axis_lims, pos_ant_dir[2]*axis_lims], 'magenta')

    # find the RGB values of the jet map for the posterior/anterior axis
    cmap = plt.cm.get_cmap('jet')
    rgba = cmap(pos_ant_loading)
    rgb = rgba[:,0:3]

    # plot the voxels (colour coded)
    ax.scatter3D(orig_coords[0],
                 orig_coords[1],
                 orig_coords[2],
                 color=rgb)

    # set a few parameters
    fig.set_facecolor('black')
    ax.set_facecolor('black')
    ax.set_xlabel('X')
    ax.set_xlim(-axis_lims, axis_lims)
    ax.xaxis.label.set_color('white')
    ax.set_ylabel('Y')
    ax.set_ylim(-axis_lims, axis_lims)
    ax.yaxis.label.set_color('white')
    ax.set_zlabel('Z')
    ax.set_zlim(-axis_lims, axis_lims)
    ax.zaxis.label.set_color('white')
    ax.tick_params(colors='white')

    return fig

