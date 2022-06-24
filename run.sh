#!/bin/bash

INPUT=${1:-imgs/input/scenes7.png}
NUM_SEGMENTS=${2:-100}
SEMANTIC=${3:-imgs/semantic.png}

echo -e "---------------\ninput RGB image: \$1=$INPUT\nnum segments: \$2=$NUM_SEGMENTS\noutput semantic RGB image: \$3=$SEMANTIC\n---------------"  
python3 semantic/get_semantic.py $INPUT $SEMANTIC
python3 slic/update_semantic.py $INPUT $SEMANTIC $NUM_SEGMENTS
python3 slic/get_mask.py imgs/slic_semantic_label.png