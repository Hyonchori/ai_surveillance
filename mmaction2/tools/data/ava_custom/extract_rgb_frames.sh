#!/usr/bin/env bash

cd ../
python build_rawframes.py /media/daton/Data/datasets/ava/videos_15min/ /media/daton/Data/datasets/ava/rawframes/ --task rgb --level 1 --mixed-ext --use-opencv
echo "Genearte raw frames (RGB only)"

cd ava/
