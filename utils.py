import sys
from PIL import Image  # pip install Pillow
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt
from random import shuffle, random
import os

def load_image(file_name):
    img = np.array(Image.open(file_name))
    print("'{}' loaded with shape = {}".format(file_name, img.shape))
    return img

def save_image(img, file_name):
    Image.fromarray(img).save(file_name)
    
def reconstruct_image(pieces_order, pieces):
    # build columns and them assemble them
    columns = []
    n_cols = pieces_order.shape[1]
    for col in range(n_cols):
        ids_in_col = pieces_order[:, col]
        temp_col = pieces[ids_in_col[0]]
        for idx in ids_in_col[1:]:
            temp_col = np.concatenate((temp_col, pieces[idx]), axis=0)
        columns.append(temp_col)
    res = columns[0]
    for i in range(1, len(columns)):
        res = np.concatenate((res, columns[i]), axis=1)
    return res

def compute_lr_border_cost(s1, s2):
    # compute a fitness score between the right part of s1 and the left part of s2
    r_col = s1[:, -1, :].astype(np.int16)
    l_col = s2[:, 0, :].astype(np.int16)
    diff_col = np.absolute(np.subtract(r_col, l_col))
    return np.sum(diff_col)

def find_n_cols(img, threshold):
    # assuming a constant column width
    costs = []
    ids = []
    print("width = img.shape[1] = {}".format(img.shape[1]))
    for i in range(img.shape[1]-1):
        l = img[:, i:i+1, :]
        r = img[:, i+1:i+2, :]
        cost = compute_lr_border_cost(l, r)
        costs.append(cost)
    for i, c in enumerate(costs):
        if c > threshold:
            ids.append(i)
    print("{} transitions above threshold={} @ pixel {}".format(len(ids), threshold, ids))
    plot_costs(costs, threshold)
    if len(ids) == 0:
        return 1
    if len(ids) == 1:
        col_width = min(img.shape[1] - ids[0], ids[0])
        print(img.shape[1] / col_width)
        res = round(img.shape[1] / col_width)
        return res
    gaps = [ids[i+1]-ids[i] for i in range(len(ids)-1)]   
    print("gaps = {}".format(gaps))
    col_width = smallest_common_diviser(gaps)
    res = int(img.shape[1] / col_width)
    return res

def find_n_rows(img, threshold):
    copy_img = np.copy(img)
    rot_img = np.copy(np.rot90(copy_img))
    return find_n_cols(rot_img, threshold)

def find_first_piece(lr_costs_matrix, ud_costs_matrix):
    # which one to go at the top-left corner
    lr_scores = np.min(lr_costs_matrix, axis=0)
    ud_scores = np.min(ud_costs_matrix, axis=0)
    tot_scores = lr_scores + ud_scores
    id_top_right_corner = np.argmax(tot_scores)
    return id_top_right_corner

def get_pieces(path):
    
    pieces = []
    for piece_path in os.listdir(path):
        img_path = path+piece_path
        if img_path == './pieces/.ipynb_checkpoints':
            pass
        else :
            piece = load_image(path+piece_path)
            pieces.append(piece)
    return pieces