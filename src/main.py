# Import necessary packages
import os
import sys
import time
import numpy as np
from tqdm import tqdm
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import vgg19
from tensorflow.keras import Model
from tensorflow_addons.layers import InstanceNormalization
import argparse

parser = argparse.ArgumentParser(description='Neural Style Transfer')

parser.add_argument('content_path', metavar='content_path', type=str,
                    help='path to the content image')
parser.add_argument('style_path', metavar='style_path', type=str,
                    help='path to the style image')
parser.add_argument('output_path', metavar='output_path', type=str,
                    help='path to the output image')
parser.add_argument('--norm', dest='norm', action='store_true',
                    help='use instance normalization')

args = parser.parse_args()

# access the arguments
content_path = args.content_path
style_path = args.style_path
output_path = os.path.splitext(args.output_path)[0] + '_{}.jpg'
instance_normalization = args.norm

# Check if the script is being run directly
if __name__ == '__main__':
    # Check if correct number of arguments are provided
    if len(sys.argv) < 4:
        print("Usage: python test.py --norm [content_path] [style_path] [content_path]")
        sys.exit(1)
    print(f"Instance normalization: {instance_normalization}")
    print(f"Content path: {content_path}")
    print(f"Style path: {style_path}")
    print(f"Output path: {output_path}")

# Define content and style layers for VGG model
content_layers = ['block5_conv2']
style_layers = [
    'block1_conv1',
    'block2_conv1',
    'block3_conv1',
    'block4_conv1',
    'block5_conv1',
]

# Define function to preprocess input image
def preprocess_image(image_path):
    # Load image from path
    img = load_img(image_path)
    # Convert image to numpy array
    img = img_to_array(img)
    # Add an extra dimension to the array
    img = np.expand_dims(img, axis=0)
    # Preprocess the image using VGG19 model preprocessing function
    img = vgg19.preprocess_input(img)
    return img

# Define function to deprocess output image
def deprocess_image(x):
    # Remove the extra dimension from the array
    x = np.squeeze(x)
    # Add the mean pixel values for VGG19 model
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # Reverse the color channels
    x = x[:, :, ::-1]
    # Clip pixel values to between 0 and 255 and convert to uint8
    x = np.clip(x, 0, 255).astype('uint8')
    return x

# Define function to get VGG19 model with specified content and style layers
def get_model(content_layers, style_layers):
    # Load VGG19 model with imagenet weights and without top layer
    vgg = vgg19.VGG19(include_top=False, weights='imagenet')
    # Freeze the model weights
    vgg.trainable = False
    content_outputs = [vgg.get_layer(name).output for name in content_layers]
    style_outputs = [vgg.get_layer(name).output for name in style_layers]

    #Add instance normalization layers if the flag is set to True
    if instance_normalization:
        # Add instance normalization layers
        for i in range(len(vgg.layers)):
            # If the layer is a convolutional layer, add an instance normalization layer after it
            if 'conv' in vgg.layers[i].name:
                norm_layer = InstanceNormalization(axis=-1)
                vgg.layers.insert(i+1, norm_layer)
                norm_layer.build(vgg.layers[i].output_shape)
                vgg.layers[i+1] = norm_layer

    # Combine the content and style outputs
    model_outputs = content_outputs + style_outputs
    # Create a Keras model with the specified inputs and outputs
    return Model(inputs=vgg.input, outputs=model_outputs)

# Define function to calculate content loss between base content and target content
def get_content_loss(base_content, target):
    return tf.reduce_mean(tf.square(base_content - target))

# Define function to calculate gram matrix of input tensor
def gram_matrix(input_tensor):
    # Add an extra dimension to the tensor
    input_tensor = tf.expand_dims(input_tensor, axis=0)
    # Calculate gram matrix using einsum function of tensorflow
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    # Calculate number of locations in input tensor
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    # Normalize the gram matrix by dividing by number of locations
    return result / num_locations

# Define function to calculate style loss between base style and target style
def get_style_loss(base_style, gram_target):
    # Calculate gram matrix of base style
    gram_style = gram_matrix(base_style)
    # Calculate mean squared error between gram matrices of base style and target style
    return tf.reduce_mean(tf.square(gram_style - gram_target))

# Define function to compute total loss for given inputs
def compute_loss(model, loss_weights, init_image, gram_style_features, content_features):
    style_weight, content_weight = loss_weights
    # Get model outputs for initial input image
    model_outputs = model(init_image)
    # Separate content and style outputs
    content_outputs = model_outputs[:len(content_features)]
    style_outputs = model_outputs[len(content_features):]

    content_score = 0
    style_score = 0

    # Calculate content loss for each content layer and add to total content score
    weight_per_content_layer = 1.0 / float(len(content_features))
    for target_content, comb_content in zip(content_features, content_outputs):
        content_score += weight_per_content_layer * get_content_loss(comb_content[0], target_content)

    # Calculate style loss for each style layer and add to total style score
    weight_per_style_layer = 1.0 / float(len(gram_style_features))
    for target_style, comb_style in zip(gram_style_features, style_outputs):
        style_score += weight_per_style_layer * get_style_loss(comb_style[0], target_style)

    # Weight content and style scores and add to get total loss
    content_score *= content_weight
    style_score *= style_weight

    loss = content_score + style_score
    return loss, content_score, style_score

# Define function to compute gradients and loss for given configuration
def compute_grads(cfg):
    with tf.GradientTape() as tape:
        all_loss = compute_loss(**cfg)
    total_loss = all_loss[0]
    # Calculate gradients of total loss with respect to initial image
    return tape.gradient(total_loss, cfg['init_image']), all_loss

# Define function to run style transfer for given input images and parameters
def run_style_transfer(content_path, style_path, num_iterations=1000, content_weight=1e3, style_weight=1e-2):
    # Get VGG19 model with specified content and style layers
    model = get_model(content_layers, style_layers)
    model.summary()
    # Freeze the model layers
    for layer in model.layers:
        layer.trainable = False
    
    # Preprocess content and style images
    content_image = preprocess_image(content_path)
    style_image = preprocess_image(style_path)
    
    # Get model outputs for content and style images
    content_outputs = model(content_image)
    style_outputs = model(style_image)
    
    # Get content and style features from model outputs
    content_features = [content_layer[0] for content_layer in content_outputs[:len(content_layers)]]
    style_features = [style_layer[0] for style_layer in style_outputs[len(content_layers):]]
    
    # Get gram matrices of style features
    gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]
    
    # Initialize initial image as content image
    init_image = tf.Variable(content_image, dtype=tf.float32)
    # Define optimizer
    opt = tf.keras.optimizers.legacy.Adam(learning_rate=5, beta_1=0.99, epsilon=1e-1)

    iter_count = 1
    best_loss, best_img = float('inf'), None
    loss_weights = (style_weight, content_weight)
    cfg = {
        'model': model,
        'loss_weights': loss_weights,
        'init_image': init_image,
        'gram_style_features': gram_style_features,
        'content_features': content_features
    }

    ratio = 20
    display_interval = num_iterations / (ratio)
    global_start = time.time()

    # Run optimization loop for specified number of iterations
    for i in tqdm(range(num_iterations)):
        # Compute gradients and loss for given configuration
        grads, all_loss = compute_grads(cfg)
        loss, content_loss, style_loss = all_loss
        # Update initial image with calculated gradients
        opt.apply_gradients([(grads, init_image)])
        # Clip pixel values to between 0 and 255
        clipped = tf.clip_by_value(init_image, 0.0, 255.0)
        init_image.assign(clipped)

        # Keep track of best loss and output image
        if loss < best_loss:
            best_loss = loss
            best_img = deprocess_image(init_image.numpy())

        # Display progress and output image at regular intervals
        if i % display_interval == 0:
            print('Iteration: {}'.format(i))
            print('Total Loss: {:.4e}, Style Loss: {:.4e}, Content Loss: {:.4e}'.format(loss, style_loss, content_loss))
            Image.fromarray(best_img).save(output_path.format(i))
            print()

    # Save final output image
    Image.fromarray(best_img).save(output_path.format(num_iterations))

    # Return final output image
    return best_img

# Get user input for num_iterations, content_weight, and style_weight
num_iterations = int(input("Enter number of iterations (default=1000): ") or "1000")
content_weight = float(input("Enter content weight (default=1000): ") or "1000")
style_weight = float(input("Enter style weight (default=0.01): ") or "0.01")

# Run style transfer with user input parameters
result_image = run_style_transfer(content_path, style_path, num_iterations, content_weight, style_weight)
# Display final output image
Image.fromarray(result_image).show()