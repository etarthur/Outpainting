# Enhanced Residual Context-based Networks for Image Outpainting 
Our work builds upon the context encoder baseline model for image outpainting proposed in 
[Image Outpaintng and Harmonization using Generative Adversarial Networks](https://arxiv.org/abs/1912.10960). This project was for the class Deep Learning by Professor Jacob Whitehill at Worcester Polytechnic Institute.

## Summary
We generate a 192x192 image from the given ground truth of the same size, masked to only show 128x128 of the target. We qualitatively evaluate improvements to the generative network and discriminator including implementing super-resolution upscaling techniques.

## Examples
![Example of outpainting models](/tex/figs/fig2/fig2_constructed.png)

## Usage
Our models are separated in their respective folders, but each use a [train](http://data.csail.mit.edu/places/places365/train_256_places365standard.tar) and [val](http://data.csail.mit.edu/places/places365/val_256.tar) folder in the repository root trainings, which contain the MIT Places365-Standard dataset.
* Run `train.py` of each model to train the network
* Evaluate custom input image by running `forward.py input.jpg output.jpg` 
