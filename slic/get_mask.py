import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import sys, os

assert os.path.isdir("imgs/")

slic_semantic_label_name = sys.argv[1]# "slic_semantic_label.png"
mask_dir = "imgs/mask/"


slic_semantic_label = cv2.imread(slic_semantic_label_name, -1)
if slic_semantic_label is None: raise FileNotFoundError("cannot found img")
h, w = slic_semantic_label.shape# grayscale image, so channel is 1, not 3

lis = []
for i in range(h):
	for j in range(w):
		k = slic_semantic_label[i][j]#class label
		# print(k)
		lis.append(k)

sets = set(lis)
val_lis = list(sets)
print(val_lis)


lis_segment = []
for i in range(len(val_lis)):
  Lower = np.array(val_lis[i])
  Upper = np.array(val_lis[i])
  tmp_mask = cv2.inRange(slic_semantic_label, Lower, Upper) # grayscale image-->マスクを作成
  # result = cv2.bitwise_and(dst, dst, mask=tmp_mask) # 元画像とマスクを合成
  cv2.imwrite(mask_dir + "mask"+str(i)+".png", tmp_mask)
  # lis_segment.append(result)