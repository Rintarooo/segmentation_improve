# https://drive.google.com/drive/folders/19Gz8u2tUZnVrQ1XVZyX_0ppJUcq3UHdY
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
# Much faster than the standard class
from fast_slic.avx2 import SlicAvx2
from skimage.segmentation import mark_boundaries
import cv2
import sys, os

assert os.path.isdir("imgs/")

rgb_name = sys.argv[1]#"cur.png"
semantic_rgb_name = sys.argv[2]#"semantic.png"
slic_semantic_rgb_name = "imgs/slic_semantic_rgb.png"#sys.argv[3]
# how many segments super pixels divide?
num_segments = int(sys.argv[3])#35#125#75#100
# fast_slic_name = os.path.dirname(__file__) + '/' + "FastSLIC_%d.png" % (num_segments)
fast_slic_name = "imgs/FastSLIC_%d.png" % (num_segments)
slic_semantic_label_name = "imgs/slic_semantic_label.png"

with Image.open(rgb_name) as f:
   rgb = np.array(f)

plt.imshow(rgb)
plt.title(rgb_name)
plt.show()


slic = SlicAvx2(num_components=num_segments, compactness=10)
slic_label = slic.iterate(rgb) # Cluster Map
slic_label += 1# [0, num_segments-1] into [1, num_segments]

# print(slic_label)
# print(slic.slic_model.clusters) # The cluster information of superpixels.


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_title("FastSLIC %d segments" % (num_segments))
ax.imshow(mark_boundaries(rgb, slic_label))
plt.axis("off")
plt.show()

visu = mark_boundaries(rgb, slic_label)*255
visu = visu.astype(np.uint8)
# fast_slic_name = os.path.dirname(__file__) + '/' + "FastSLIC_%d.png" % (num_segments)
cv2.imwrite(fast_slic_name, cv2.cvtColor(visu, cv2.COLOR_RGB2BGR))
print("save image: " + fast_slic_name)

semantic_rgb = cv2.imread(semantic_rgb_name)
if semantic_rgb is None: raise FileNotFoundError("cannot found img")
# print("semantic_rgb.shape: ", semantic_rgb.shape)
h, w, _ = semantic_rgb.shape
# https://python.blueskyzz.com/opencv/imread_write/
# cv2.imshow()
# plt.imshow(cv2.cvtColor(semantic_rgb, cv2.COLOR_BGR2RGB))
# plt.title(semantic_rgb_name)
# plt.show()


lis = []
for i in range(h):
  for j in range(w):
    tup = tuple(semantic_rgb[i][j])
    lis.append(tup)

b = set(lis)
dst = list(b)
num_class = len(dst)
print("semantic num_class: ", num_class)

semantic_rgb_dic = {k:dst[k-1] for k in range(1, num_class+1)}
print("semantic_rgb_dic(key: class label, val: rgb color)", semantic_rgb_dic)

semantic_label = np.zeros((h,w))
for i in range(h):
  for j in range(w):
    rgb_val = tuple(semantic_rgb[i][j])
    for k in range(1, num_class+1):
      if(rgb_val==semantic_rgb_dic[k]):
        semantic_label[i][j] = k#class label

semantic_rgb = np.zeros((h,w,3))
for i in range(h):
  for j in range(w):
    label_idx = semantic_label[i][j]
    if(label_idx==0): print(i, j)
    else:
      rgb_tuple = semantic_rgb_dic[label_idx]
      semantic_rgb[i][j]=rgb_tuple

semantic_rgb = semantic_rgb.astype(np.uint8)


slic_semantic_label = np.zeros((h,w))
for k in range(1, num_segments+1):
  print(k)
  tmp = np.zeros((h,w))
  slic_each_label = slic_label.copy()
  # slic_each_label += 1# [0,num_segment-1] into [1,num_segment]
  slic_each_label[slic_each_label != k] = 0
  slic_each_label[slic_each_label == k] = 1
  tmp = slic_each_label * semantic_label
  tmp = tmp.astype(np.uint8)
  # print("np.bincount(tmp.flatten())[1:]:", np.bincount(tmp.flatten())[1:])
  if(len(np.bincount(tmp.flatten())[1:])==0): continue
  occupy_idx = np.argmax(np.bincount(tmp.flatten())[1:])+1#[1:] means except 0
  
  # print("most occupied class label index in %d segement of %d segments from super pixels(fast slic) is %d in semantic label[1, %d]" % (k, num_segments, occupy_idx, class_num))
  
  tmp[tmp > 0] = occupy_idx
  slic_semantic_label += tmp

print("slic_semantic_label: ", slic_semantic_label)

scale_img = slic_semantic_label*255/num_class# normalize [0,8] into [0,255], for visualizing
plt.imshow(scale_img)
plt.title("semantic label + super pixels label")
plt.show()
# cv2.imwrite(os.path.dirname(__file__) + '/' + "slic_semantic_label.png", scale_img)
cv2.imwrite(slic_semantic_label_name, scale_img)



slic_semantic_rgb = np.zeros((h,w,3))
for i in range(h):
  for j in range(w):
    label_idx = slic_semantic_label[i][j]
    if(label_idx==0): print("debug: ", i, j)
    else:
      rgb_tuple = semantic_rgb_dic[label_idx]
      slic_semantic_rgb[i][j]=rgb_tuple

slic_semantic_rgb = slic_semantic_rgb.astype(np.uint8)
# print(semantic_rgb)
# plt.imshow(semantic_rgb)
plt.imshow(cv2.cvtColor(slic_semantic_rgb, cv2.COLOR_BGR2RGB))
plt.title("semantic + super pixels label --> semantic RGB")
plt.show()
# cv2.imwrite("slic_semantic_rgb.png", slic_semantic_rgb)
cv2.imwrite(slic_semantic_rgb_name, slic_semantic_rgb)