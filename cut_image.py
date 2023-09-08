import sys
from PIL import Image  # pip install Pillow
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt
from random import shuffle, random
import argparse
from utils import load_image, save_image, reconstruct_image


def shuffle_image(img, n_cols, n_rows,new_path,transform=False):
    def random_transform(image):
        # Apply random transforms (flip, mirror, rotate)
        if random() < 0.5:
            image = np.flip(image, axis=0)  # Vertical flip
            print('mirror')
        if random() < 0.5:
            image = np.flip(image, axis=1)  # Horizontal flip
            print('flip')
        if random() < 0.5:
            image = np.rot90(image, k=-1, axes=(0, 1))  # 90-degree rotation
            print('rotate')

        return image
    print(n_cols)
    # First generate a list of piece indices, then build the corresponding image
    row_height = int(img.shape[0] / n_rows)
    col_width = int(img.shape[1] / n_cols)
    rows = [img[row_height*i:row_height*(i+1), :, :] for i in range(n_rows)]

    pieces = []
    for row in rows:
        for col_id in range(n_cols):
            piece = row[:, col_width*col_id:col_width*(col_id+1):, :]
            if transform :
                piece = random_transform(piece)  # Apply random transforms
            pieces.append(piece)

    random_order = list(range(n_cols*n_rows))
    shuffle(random_order)  # Equivalent to random.shuffle
    random_order = np.array(random_order).reshape(n_rows, n_cols)
    print("Random order = {}".format(random_order))
    print("Len(pieces) = {}".format(len(pieces)))
    
    for i in range(len(pieces)) :
        image = Image.fromarray(pieces[i])
        image.save(new_path+str(i)+'.png')
        
        
    #return reconstruct_image(random_order, pieces)
    return random_order, pieces


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Cut an image into a grid.")

    parser.add_argument("--column_num", type=int, help="Number of columns")
    parser.add_argument("--row_num", type=int, help="Number of rows")
    parser.add_argument("--prefix_output_file_name", help="Prefix for output file names")
    parser.add_argument("--transform", help="Prefix for output file names")
    

    args = parser.parse_args()
    print(f"Column Num: {args.column_num}")
    print(f"Row Num: {args.row_num}")
    print(f"Prefix Output File Name: {args.prefix_output_file_name}")
    image_pth = 'original.png'
    image = load_image(image_pth)

    _,_ = shuffle_image(image,args.column_num,args.row_num,args.prefix_output_file_name)

