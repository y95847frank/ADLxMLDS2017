#!/bin/sh

wget https://www.csie.ntu.edu.tw/~b03902052/best.h5 -P model/
python load_model.py model/best.json model/best.h5 $2 $1
