#!/bin/bash

uv sync

# Get weights
if [ -d "checkpoints" ]; then
    echo "El directorio checkpoints ya existe, omitiendo descarga de pesos";
else
    mkdir -p checkpoints;
    wget https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt -P checkpoints;
fi

# Get data
cat data/train.txt | xargs -n 1 -P 4 wget -P data/train;
cat data/test.txt | xargs -n 1 -P 4 wget -P data/test;
unzip "data/train/*.zip" -d data/train;
unzip "data/test/*.zip" -d data/test;
rm data/train/*.zip;
rm data/test/*.zip;
find data -type f ! -name "*.color.jpg" ! -name "*.est.npy" ! -name "*.depth_meters.hdf5" -print0 | xargs -0 rm --

# Preprocess data
uv run preprocess
