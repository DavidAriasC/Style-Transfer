import os
import sys
import time
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import vgg19
from tensorflow.keras import Model

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python test.py [content_path] [style_path] [content_path]")
        sys.exit(1)
    content_path = sys.argv[1]
    style_path = sys.argv[2]
    output_path = sys.argv[3]
    output_path = os.path.splitext(output_path)[0] + '_{}.jpg'
    print(f"Content path: {content_path}")
    print(f"Style path: {style_path}")
    print(f"Output path: {output_path}")

content_layers = ['block5_conv2']
style_layers = [
    'block1_conv1',
    'block2_conv1',
    'block3_conv1',
    'block4_conv1',
    'block5_conv1',
]

def preprocess_image(image_path):
    img = load_img(image_path)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img

def deprocess_image(x):
    x = np.squeeze(x)  # Add this line to fix the issue
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def get_model(content_layers, style_layers):
    vgg = vgg19.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    content_outputs = [vgg.get_layer(name).output for name in content_layers]
    style_outputs = [vgg.get_layer(name).output for name in style_layers]
    model_outputs = content_outputs + style_outputs
    return Model(inputs=vgg.input, outputs=model_outputs)

def get_content_loss(base_content, target):
    return tf.reduce_mean(tf.square(base_content - target))

def gram_matrix(input_tensor):
    input_tensor = tf.expand_dims(input_tensor, axis=0)  # Add this line to fix the issue
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations

def get_style_loss(base_style, gram_target):
    gram_style = gram_matrix(base_style)
    return tf.reduce_mean(tf.square(gram_style - gram_target))

def compute_loss(model, loss_weights, init_image, gram_style_features, content_features):
    style_weight, content_weight = loss_weights
    model_outputs = model(init_image)
    content_outputs = model_outputs[:len(content_features)]
    style_outputs = model_outputs[len(content_features):]

    content_score = 0
    style_score = 0

    weight_per_content_layer = 1.0 / float(len(content_features))
    for target_content, comb_content in zip(content_features, content_outputs):
        content_score += weight_per_content_layer * get_content_loss(comb_content[0], target_content)

    weight_per_style_layer = 1.0 / float(len(gram_style_features))
    for target_style, comb_style in zip(gram_style_features, style_outputs):
        style_score += weight_per_style_layer * get_style_loss(comb_style[0], target_style)

    content_score *= content_weight
    style_score *= style_weight

    loss = content_score + style_score
    return loss, content_score, style_score

def compute_grads(cfg):
    with tf.GradientTape() as tape:
        all_loss = compute_loss(**cfg)
    total_loss = all_loss[0]
    return tape.gradient(total_loss, cfg['init_image']), all_loss

def run_style_transfer(content_path, style_path, num_iterations=1000, content_weight=1e3, style_weight=1e-2):
    model = get_model(content_layers, style_layers)
    model.summary()
    for layer in model.layers:
        layer.trainable = False
    
    content_image = preprocess_image(content_path)
    style_image = preprocess_image(style_path)
    
    content_outputs = model(content_image)
    style_outputs = model(style_image)
    
    content_features = [content_layer[0] for content_layer in content_outputs[:len(content_layers)]]
    style_features = [style_layer[0] for style_layer in style_outputs[len(content_layers):]]
    
    gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]
    
    init_image = tf.Variable(content_image, dtype=tf.float32)
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
    # start_time = time.time()
    global_start = time.time()

    for i in range(num_iterations):
        grads, all_loss = compute_grads(cfg)
        loss, content_loss, style_loss = all_loss
        opt.apply_gradients([(grads, init_image)])
        clipped = tf.clip_by_value(init_image, 0.0, 255.0)
        init_image.assign(clipped)

        if loss < best_loss:
            best_loss = loss
            best_img = deprocess_image(init_image.numpy())

        if i % display_interval == 0:
            print('Iteration: {}'.format(i))
            print('Total Loss: {:.4e}, Style Loss: {:.4e}, Content Loss: {:.4e}'.format(loss, style_loss, content_loss))
            Image.fromarray(best_img).save(output_path.format(i))
            print()

    print('Total time: {:.4f}s'.format(time.time() - global_start))
    Image.fromarray(best_img).save(output_path.format(num_iterations))
    return best_img

# Get user input for num_iterations, content_weight, and style_weight
num_iterations = int(input("Enter number of iterations (default=1000): ") or "1000")
content_weight = float(input("Enter content weight (default=1000): ") or "1000")
style_weight = float(input("Enter style weight (default=0.01): ") or "0.01")

result_image = run_style_transfer(content_path, style_path, num_iterations, content_weight, style_weight)
Image.fromarray(result_image).show()