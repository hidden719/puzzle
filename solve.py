import sys
from PIL import Image  # pip install Pillow
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt
from random import shuffle, random
import argparse
from utils import *

class Solver:
    # lr = left-right
    # td = top-down
    def __init__(self, lr_costs_matrix, td_costs_matrix, n_cols, n_rows, first_piece):
        self.lr_cost_matrix = np.copy(lr_costs_matrix)
        self.td_cost_matrix = np.copy(td_costs_matrix)
        self.n_cols = n_cols
        self.n_rows = n_rows
        self.first_piece = first_piece
        self.used_ids = []
        self.res_in_levels = []
    
    def get_idx(self, level, row):
        assert level >= row
        col = level - row
        idx = row * self.n_cols + col
        return idx

    def get_idx_parents(self, level, row):
        # parents have level-1
        # parent_l has col-1 and same row
        # parent_t has same col and row-1
        if level == 0:
            return None, None
        parents_level = level - 1
        if row == level:  # child is on left-most column 
            return None, self.get_idx(parents_level, row-1)
        if row == 0:  # child is on top-most column
            return self.get_idx(parents_level, 0), None
        parent_l_row = row
        parent_t_row = row - 1
        parent_l = self.get_idx(parents_level, parent_l_row)
        parent_t = self.get_idx(parents_level, parent_t_row)
        return parent_l, parent_t
    
    def find_best(self, parent_l, parent_t):
        r_candidates = self.lr_cost_matrix[parent_l, :] if parent_l is not None else 0
        d_candidates = self.td_cost_matrix[parent_t, :] if parent_t is not None else 0
        total_candidates = r_candidates + d_candidates
        found = False
        while not found:
            if np.min(total_candidates) == np.inf:
                print("ERROR: no piece found after self.res_in_levels = [{}]".format(self.res_in_levels))
                return None
            best = np.argmin(total_candidates)
            if best in self.used_ids:
                print("ISSUE: want to add {} a second time as child of parent_l = {} and parent_t = {}".format(best, parent_l, parent_t))
                total_candidates[best] = np.inf
            else:
                found = True
                self.used_ids.append(best)
        return best
    
    def format_res_levels(self, res_in_levels):
        res = np.zeros((self.n_rows, self.n_cols))
        for row in range(self.n_rows):
            for col in range(self.n_cols):
                level = row + col
                res[row, col] = res_in_levels[level][row]
        print("res_in_levels = {}".format(res_in_levels))
        print("format_levels = {}".format(res))
        return res.astype(np.int16)

    def get_id_in_res(self, position_in_flat_list, res_in_levels):
        row = position_in_flat_list // self.n_cols
        col = position_in_flat_list % self.n_cols
        level = row + col
        piece_at_position = res_in_levels[level][row]
        return piece_at_position
    
    def solve(self):
        self.lr_cost_matrix[:, self.first_piece] = np.inf  # first piece cannot be a right-neighbour
        self.td_cost_matrix[:, self.first_piece] = np.inf  # first piece cannot be a down-neighbour
        self.used_ids.append(self.first_piece)
        self.res_in_levels = [[first_piece]]  # alone in level=0
        
        for level in range(1, self.n_cols + self.n_rows - 1):
            list_for_level = []
            for row_id in range(min(self.n_rows, level+1)):  # no need to explore row > n_rows
                if level + 1 - row_id > self.n_cols:
                    list_for_level.append(None)
                else:
                    # position of the parents in the 0, 1, 2 ... coordinate system
                    parent_l_out, parent_t_out = self.get_idx_parents(level, row_id)
                    parent_l_in = self.get_id_in_res(parent_l_out, self.res_in_levels) if parent_l_out is not None else None
                    parent_t_in = self.get_id_in_res(parent_t_out, self.res_in_levels) if parent_t_out is not None else None
                    best = self.find_best(parent_l_in, parent_t_in)
                    list_for_level.append(best)
            self.res_in_levels.append(list_for_level)
        res = self.format_res_levels(self.res_in_levels)
        return res
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Cut an image into a grid.")

    parser.add_argument("--column_num", type=int, help="Number of columns")
    parser.add_argument("--row_num", type=int, help="Number of rows")
    parser.add_argument("--result", help="Prefix for output file names")

    args = parser.parse_args()
    
    pieces = get_pieces("./pieces/")
    
    n_cols = args.column_num
    n_rows = args.row_num
    
        
    n_pieces = n_cols*n_rows
    lr_border_costs = np.zeros((n_pieces, n_pieces))
    td_border_costs = np.zeros((n_pieces, n_pieces))
    for p1 in range(n_pieces):
        for p2 in range(n_pieces):
            lr_border_costs[p1][p2] = np.inf if p1==p2 else compute_lr_border_cost(pieces[p1], pieces[p2])
            td_border_costs[p1][p2] = np.inf if p1==p2 else compute_lr_border_cost(np.rot90(pieces[p1]), np.rot90(pieces[p2]))
    first_piece = find_first_piece(lr_border_costs, td_border_costs)
    solver = Solver(lr_border_costs, td_border_costs, n_cols, n_rows, first_piece)
    good_order = solver.solve()
    reconstructed_img = reconstruct_image(good_order, pieces)    
    save_image(reconstructed_img, args.result+'.png')    