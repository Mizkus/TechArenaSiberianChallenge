#!/bin/bash
URL="https://huggingface.co/datasets/fzliu/sift1m/resolve/main/sift.tar.gz?download=true"
TARGET_DIR="sift"

mkdir -p "$TARGET_DIR"
wget -O sift.tar.gz "$URL"
unzip sift.tar.gz -d "$TARGET_DIR"
rm sift.tar.gz

echo "Dataset SIFT1M succesfully download and unpacked in folder '$TARGET_DIR'"
