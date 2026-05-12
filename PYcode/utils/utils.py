import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import os


def readSWC(swc_path, use_bouton=False):
    n_skip = 0
    with open(swc_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                n_skip += 1
            else:
                break

    if not use_bouton:
        df = pd.read_csv(
            swc_path,
            index_col=0,
            skiprows=n_skip,
            sep=r"\s+",
            engine="python",
            usecols=[0, 1, 2, 3, 4, 5, 6],
            names=["##n", "type", "x", "y", "z", "r", "parent"],
        )
        for col in ["type", "x", "y", "z", "r", "parent"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["type", "x", "y", "z", "parent"])
        df["type"] = df["type"].astype(np.int32)
        df["parent"] = df["parent"].astype(np.int64)
    else:
        df = pd.read_csv(
            swc_path,
            skiprows=n_skip,
            sep=r"\s+",
            engine="python",
            usecols=[0, 1, 2],
            names=["x", "y", "z"],
        )
        for col in ["x", "y", "z"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["x", "y", "z"])

    return df


def func(swc1, swc2, dest, use_bouton=True, swc_id1=None, swc_id2=None):
    if use_bouton:
        axon1 = swc1[['x', 'y', 'z']]
        den2 = swc2.loc[~swc2.type.isin([1, 2, 5]), ['x', 'y', 'z']]  # swc2.type.isin([3, 4])
    else:
        axon1 = swc1.loc[swc1.type.isin([2]), ['x', 'y', 'z']]
        den2 = swc2.loc[~swc2.type.isin([1, 2, 5]), ['x', 'y', 'z']]  # swc2.type.isin([3, 4])
    axon1 = axon1.astype(int)
    den2 = den2.astype(int)

    # den_x_min = np.min(den2['x'])
    # den_x_max = np.max(den2['x'])
    #
    # den_y_min = np.min(den2['y'])
    # den_y_max = np.max(den2['y'])
    #
    # den_z_min = np.min(den2['z'])
    # den_z_max = np.max(den2['z'])
    #
    # if (len(np.argwhere(np.abs(axon1['x'] - den_x_max).values < 10)) == 0) and \
    #         (len(np.argwhere(np.abs(axon1['x'] - den_x_min).values < 10)) == 0):
    #     os.system('touch ' + dest.replace('results', 'no_results') + '/' + swc_id1 + '_' + swc_id2 + '.txt')
    #     return
    #
    # if (len(np.argwhere(np.abs(axon1['y'] - den_y_max).values < 10)) == 0) and \
    #         (len(np.argwhere(np.abs(axon1['y'] - den_y_min).values < 10)) == 0):
    #     os.system('touch ' + dest.replace('results', 'no_results') + '/' + swc_id1 + '_' + swc_id2 + '.txt')
    #     return
    #
    # if (len(np.argwhere(np.abs(axon1['z'] - den_z_max).values < 10)) == 0) and \
    #         (len(np.argwhere(np.abs(axon1['z'] - den_z_min).values < 10)) == 0):
    #     os.system('touch ' + dest.replace('results', 'no_results') + '/' + swc_id1 + '_' + swc_id2 + '.txt')
    #     return

    a2d_values = cdist(axon1.values, den2.values, 'sqeuclidean')
    a2d_index = np.argwhere(a2d_values <= 25)

    if len(a2d_index) > 0:
        pd.DataFrame({'axon_id': axon1.index[a2d_index[:, 0]],
                      'den_id': den2.index[a2d_index[:, 1]],
                      'dis': a2d_values[a2d_index[:, 0], a2d_index[:, 1]]
                      }).to_csv(dest, sep=',')

    return


def connectivity(swc1, swc2, swc_id1, swc_id2, dest,
                 use_bouton=True):
    try:
        # print('reading : ', swc_id1, ' : ', swc_id2)
        func(swc1 = swc1, swc2 = swc2, dest = dest, use_bouton=use_bouton)

    except Exception as e:
        print('fail at: ', e, ' : swc: ', swc_id1, ' ', swc_id2)

    return
