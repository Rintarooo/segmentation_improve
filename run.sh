#!/bin/bash

# input: RGB image
INPUT=${1:-imgs/input/scenes7.png}
# how many segments super pixels divide?
NUM_SEGMENTS=${2:-100}
# output: semantic.png
SEMANTIC=${3:-imgs/semantic.png}

echo -e "---------------\ninput RGB image: \$1=$INPUT\nnum segments: \$2=$NUM_SEGMENTS\noutput semantic RGB image: \$3=$SEMANTIC\n---------------"  

# CNN outputs semantic segmentation RGB
# input: scenes7.png --> output: semantic.png
python3 semantic/get_semantic.py $INPUT $SEMANTIC

# convert semantic segmentation RGB into slic + semantic segmentation RGB 
# inputs: scenes7.png + semantic.png --> output: slic_semantic_rgb.png
python3 slic/update_semantic.py $INPUT $SEMANTIC $NUM_SEGMENTS

# create imgs/mask/*.png
python3 slic/get_mask.py imgs/slic_semantic_label.png