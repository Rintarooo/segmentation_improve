## Superpixel for improving Semantic Segmentation

### Usage 0(recommend)
```bash
# download weight of CNN
./download_weight.sh

# create docker container
./docker/docker.sh run

# run program
./run.sh imgs/input/scenes7.png 100 imgs/semantic.png
```


### Usage 1(non-recommend)
```bash
./docker/docker.sh run

# CNN outputs semantic segmentation RGB
# input: cur.png --> output: semantic.png
python3 semantic-segmentation-pytorch/colab.py imgs/input/cur.png imgs/semantic.png

# semantic segmentation RGB into fast slic + semantic RGB 
# inputs: cur.png + semantic.png --> output: slic_semantic_rgb.png
# how many segments super pixels divide?
python3 slic_colab/colab.py imgs/input/cur.png imgs/semantic.png 35


# inputs: slic_semantic_rgb.png --> output: imgs/mask/*.png
python3 slic_colab/get_mask.py imgs/slic_semantic_label.png
```

## Reference
- Semantic Segmentaion model
	- https://github.com/CSAILVision/semantic-segmentation-pytorch
- Superpixel(FastSLIC)
	- https://github.com/Algy/fast-slic
