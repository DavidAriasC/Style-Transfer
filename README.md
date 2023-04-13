## Neural Style Transfer

This code uses Neural Style Transfer to blend the content of an image with the style of another image, producing a new image that represents the content of the first image with the style of the second.

### Prerequisites

To run this code you'll need:

- Python 3
- TensorFlow 2
- Numpy
- Pillow

### Usage

To use the code, simply specify the paths to the content and style images at the top of the file:

```python
content_path = 'content.jpg'
style_path = 'style.jpg'
output_path = 'output.jpg'

# Run style transfer
run_style_transfer(content_path, style_path, num_iterations=1000, content_weight=1e3, style_weight=1e-2)

# Output image is saved at output_path
```

### Parameters
* num_iterations: The number of iterations to run the optimization for (default: 1000).
* content_weight: The weight given to the content loss (default: 1e3).
* style_weight: The weight given to the style loss (default: 1e-2).

### References
Gatys, L. A., Ecker, A. S., & Bethge, M. (2015). A Neural Algorithm of Artistic Style. arXiv preprint arXiv:1508.06576.
