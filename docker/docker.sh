#!/bin/bash

# if $2 is None
# image:tag is rin/torch:cuda-10.1-cudnn7-devel-ubuntu18.04
IMAGE_NAME=${2:-rin/torch:cuda-10.1-cudnn7-devel-ubuntu18.04}

if [ "$1" = "build" ]; then
	docker build -t $IMAGE_NAME .
	echo -e "\n\n\ndocker images | head"
	docker images | head
elif [ "$1" = "run" ]; then
	docker run -it --rm  \
		--gpus=all \
		-v ${PWD}:/app \
		$IMAGE_NAME
else
	echo "command should be either one:
	 ${0} build
	 or
	 ${0} run"
fi