
import os 
import pandas as pd
import numpy as np
import cv2

svg_ending = '.svg'
png_ending = '.png'
path  = r'C:\MSC\HC\HW1\Shepp_logan.png'
# svg_path  = path + svg_ending
# png_path  = path + png_ending

# read image 
CT_image = cv2.imread(path)
cv2.imshow('CT', CT_image)





