from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import base64
from io import BytesIO
import cv2

app = Flask(__name__)

# Load the Xception model
model = load_model('./model.h5')

# Define Grad-CAM function for ResNet50
def grad_cam(input_image, model, layer_name="conv5_block3_out"):
    """
    Apply Grad-CAM on the given input image for the Xception model.
    
    Parameters:
        input_image (Tensor): Preprocessed image tensor
        model (Model): Pre-trained Xception model
        layer_name (str): Name of the convolutional layer for Grad-CAM

    Returns:
        np.ndarray: Heatmap image
    """
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(input_image)
        loss = predictions[:, tf.argmax(predictions[0])]
    
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(image, heatmap):
    """
    Overlay the heatmap onto the original image.
    
    Parameters:
        image (PIL.Image): Original input image
        heatmap (np.ndarray): Heatmap to overlay

    Returns:
        PIL.Image: Image with overlay
    """
    heatmap = cv2.resize(heatmap, (image.size[0], image.size[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlayed_img = heatmap * 0.4 + np.array(image)
    return Image.fromarray(np.uint8(overlayed_img))

@app.route('/gradcam', methods=['POST'])
def gradcam_endpoint():
    try:
        data = request.json['imageData']
        input_array = np.array(data).reshape(224, 224, 3)
        input_tensor = tf.convert_to_tensor(input_array, dtype=tf.float32)
        input_tensor = tf.expand_dims(input_tensor, 0)  # Add batch dimension

        heatmap = grad_cam(input_tensor, model)

        # Prepare the original image for overlay
        original_image = (input_tensor[0].numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(original_image)

        overlayed_img = overlay_heatmap(pil_image, heatmap)

        # Encode overlayed image to return as base64
        buffer = BytesIO()
        overlayed_img.save(buffer, format="PNG")
        encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return jsonify({"gradCam": encoded_image})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
