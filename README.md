# Neural Style Transfer

This code uses the VGG19 architecture for neural style transfer. Given a content image and a style image, it applies the style of the style image to the content image.

## Requirements

This code requires the following packages to be installed:

- numpy
- tensorflow
- Pillow

## Usage

Run the script `test.py` and provide the path to the content image, the style image, and the output image.

```python
python test.py [content_path] [style_path] [output_path]
```

The script will generate an image with the style of the style image applied to the content image, and save it to the specified output path.

## How it works

The code loads the VGG19 architecture with pre-trained ImageNet weights. It then extracts the features of the content image and the style image from specific layers of the network. The content loss is calculated as the mean squared error between the feature maps of the content image and the generated image. The style loss is calculated as the mean squared error between the Gram matrices of the feature maps of the style image and the generated image. The total loss is a weighted sum of the content loss and the style loss. The generated image is updated using the Adam optimizer to minimize the total loss.


https://user-images.githubusercontent.com/26073311/232408336-a68b7a4e-15f3-4e26-8bf3-f88df05e451f.mp4


## Parameters
* num_iterations: The number of iterations to run the optimization for (default: 1000).
* content_weight: The weight given to the content loss (default: 1e3).
* style_weight: The weight given to the style loss (default: 1e-2).

## References
Gatys, L. A., Ecker, A. S., & Bethge, M. (2015). A Neural Algorithm of Artistic Style. arXiv preprint arXiv:1508.06576.
