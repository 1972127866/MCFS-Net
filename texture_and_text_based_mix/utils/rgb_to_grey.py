
import cv2
import os
from tqdm import tqdm

def listdir(path, list_name):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        list_name.append(file_path)

img_dir_path = r"D:\aimeelina\editGAN\huawen\textures_for_train"
save_path=r"D:\aimeelina\editGAN\huawen\textures_with_edge"
path_list = []
listdir(img_dir_path, path_list)
for image_p in tqdm(path_list):

img_gray = cv2.imread(r'D:\aimeelina\deltaEdit\output\mini_textures_for_test\mini_textures_for_test\7.jpg', flags = 0)
img2 = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
cv2.imwrite(r'D:\aimeelina\deltaEdit\output\mini_textures_for_test\mini_textures_for_test\7_grey.jpg',img2)

