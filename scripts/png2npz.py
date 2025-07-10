from PIL import Image
import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path_to_files', type=str, help='the npz dataset file')
args = parser.parse_args()
 
#path_to_files = "../scripts/imagenet_dataloader/imagenet256_clean"
if args.path_to_files[-1] == "/":
    output_path = args.path_to_files[: -1] + ".npz"
else: 
    output_path = args.path_to_files + ".npz" 

array_of_images = []
for i, file in enumerate(sorted(os.listdir(args.path_to_files))):
    if i% 100 == 99:
        print(i+1, "images gathered")
    single_im = Image.open(os.path.join(args.path_to_files,file))
    single_array = np.array(single_im)
    array_of_images.append(single_array)         
np.savez(output_path, array_of_images) # save all in one file
print("Saved as", output_path)
