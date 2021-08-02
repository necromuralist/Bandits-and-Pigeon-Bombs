#! /usr/bin/env fish
docker run --gpus all -p 2222:22 --name bandits-numba-conda \
       --mount type=bind,source=$HOME/projects/Bandits-and-Pigeon-Bombs,target=/home/neurotic/Bandits-and-Pigeon-Bombs \
       --mount type=bind,source=/media/data,target=/home/neurotic/data \
       --mount type=bind,source=$HOME/projects/graeae,target=/home/neurotic/graeae \
       --mount type=bind,source=$HOME/projects/models/,target=/home/neurotic/models \
       -it bandits-numba-conda bash
