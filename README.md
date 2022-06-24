## Superpixel for improving Semantic Segmentation

### Usage
```bash
# download weight of CNN
./download_weight.sh

# create docker container
./docker/docker.sh run

# run program
./run.sh imgs/input/scenes7.png 100 imgs/semantic.png
```


## Reference
- Semantic Segmentaion model
	- https://github.com/CSAILVision/semantic-segmentation-pytorch
- Superpixel(FastSLIC)
	- https://github.com/Algy/fast-slic
