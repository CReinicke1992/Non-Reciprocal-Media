#!/usr/bin/env
# crop_png.sh Crop white border of a png using ImageMagick

convert $1 -trim cropped/$1
