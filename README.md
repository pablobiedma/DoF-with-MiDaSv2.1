I've coded this depth.py file aiming at achieving a faster way to computer DoF data of a given image.

# Depth of Field Analysis Using MiDaS

## Overview
This script uses the MiDaS model to calculate depth information from a set of images. The script generates depth maps, providing a visual representation of depth for each image. Additionally, it computes average, maximum, and minimum depth values, which are stored and can be used for further analysis.

## Requirements
- Python 3.x
- PyTorch
- OpenCV
- Matplotlib
- NumPy

## How to Run
After using pip to install the required libraries simply change the input and output path to your desired path and run depth.py

## Examples
Below are examples of the script output showing a side-by-side display of the original images and their depth maps, with the depth statistics printed below each pair.

![Example Image 1](https://i.imgur.com/kFfO2Nt.png)
![Example Image 2](https://imgur.com/kFfO2Nt)


