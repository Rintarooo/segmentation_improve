# System libs
import os, csv, torch, numpy, scipy.io, PIL.Image, torchvision.transforms
# Our libs
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import colorEncode
# OpenCV
import cv2
import sys

rgb_name = sys.argv[1]
save_name = sys.argv[2]
assert os.path.isdir("imgs/")

colors = scipy.io.loadmat(os.path.dirname(__file__) + '/' + 'data/color150.mat')['colors']
names = {}
with open(os.path.dirname(__file__) + '/' + 'data/object150_info.csv') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        names[int(row[0])] = row[5].split(";")[0]

def visualize_result(img, pred, index=None):
    # filter prediction class if requested
    if index is not None:
        pred = pred.copy()
        pred[pred != index] = -1
        print(f'{names[index+1]}:')
        
    # colorize prediction
    pred_color = colorEncode(pred, colors).astype(numpy.uint8)
    # print("pred_color.shape: ", pred_color.shape)# (512, 768, 3)

    # save using OpenCV
    if index is not None:
      pred_gray = cv2.cvtColor(pred_color, cv2.COLOR_BGR2GRAY)
      th = 5# 閾値
      i_max = 255# 最大輝度値
      _, i_binary = cv2.threshold(pred_gray, th, i_max, cv2.THRESH_BINARY)# 二値化処理

      # cv2.imwrite("mask/" + names[index+1] + ".png", pred_color)
      # cv2.imwrite("mask/" + names[index+1] + ".png", pred_gray)
      cv2.imwrite(os.path.dirname(__file__) + '/' + "mask/" + names[index+1] + ".png", i_binary)
    else:
      # save_name = "semantic.png"
      cv2.imwrite(save_name, cv2.cvtColor(pred_color, cv2.COLOR_BGR2RGB))
      print("save: ", save_name)

    
    # aggregate images and save
    im_vis = numpy.concatenate((img, pred_color), axis=1)
    # PIL.Image.display(PIL.Image.fromarray(im_vis))
    # display(PIL.Image.fromarray(im_vis))
















# Network Builders
net_encoder = ModelBuilder.build_encoder(
    arch='resnet50dilated',
    fc_dim=2048,
    weights=os.path.dirname(__file__) + '/' + 'ckpt/ade20k-resnet50dilated-ppm_deepsup/encoder_epoch_20.pth')
net_decoder = ModelBuilder.build_decoder(
    arch='ppm_deepsup',
    fc_dim=2048,
    num_class=150,
    weights=os.path.dirname(__file__) + '/' + 'ckpt/ade20k-resnet50dilated-ppm_deepsup/decoder_epoch_20.pth',
    use_softmax=True)

crit = torch.nn.NLLLoss(ignore_index=-1)
segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
segmentation_module.eval()
segmentation_module.cuda()
















# Load and normalize one image as a singleton tensor batch
pil_to_tensor = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406], # These are RGB mean+std values
        std=[0.229, 0.224, 0.225])  # across a large photo dataset.
])
# pil_image = PIL.Image.open('ADE_val_00001519.jpg').convert('RGB')
pil_image = PIL.Image.open(rgb_name).convert('RGB')
img_original = numpy.array(pil_image)
img_data = pil_to_tensor(pil_image)
singleton_batch = {'img_data': img_data[None].cuda()}
output_size = img_data.shape[1:]











# Run the segmentation at the highest resolution.
with torch.no_grad():
    scores = segmentation_module(singleton_batch, segSize=output_size)
    
# Get the predicted scores for each pixel
_, pred = torch.max(scores, dim=1)
pred = pred.cpu()[0].numpy()
visualize_result(img_original, pred)



# # Top classes in answer
# predicted_classes = numpy.bincount(pred.flatten()).argsort()[::-1]
# for c in predicted_classes[:15]:
#     visualize_result(img_original, pred, c)




# from skimage.segmentation import mark_boundaries
# from PIL import Image 
# import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# # ax.set_title("Superpixels -- %d segments" % (160))
# ax.set_title("semantic segmentation")
# ax.imshow(mark_boundaries(img_original, pred))
# plt.axis("off")
# plt.show()
# # plt.save()
# fig.savefig("semantic.png")