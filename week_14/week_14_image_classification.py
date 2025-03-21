import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

# Load the trained model
model = tf.keras.models.load_model('CIFAR_10_tens.h5')

# Load the new image (replace 'C:/Users/gounh/Downloads/re-test-fig.jpeg' with the path to your image)
img_path = 'C:/Users/gounh/Downloads/re-test-fig.jpeg'  # Update this path to your image
img = image.load_img(img_path)  # Load the original image (no resizing)

# Preprocess the image for the model
img_resized = image.load_img(img_path, target_size=(32, 32))  # Resize to 32x32
img_array = image.img_to_array(img_resized)  # Convert to numpy array
img_array = img_array / 255.0  # Normalize pixel values to [0, 1]
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# Predict the class
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions, axis=1)  # Get the predicted class index
confidence = np.max(predictions)  # Get the confidence (probability) of the predicted class

# Map the predicted class index to the label
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
predicted_label = labels[predicted_class[0]]

# Display the original image with the predicted label and confidence
plt.figure(figsize=(8, 8))
plt.imshow(img)
plt.title(f"Predicted Label: {predicted_label}\nConfidence: {confidence:.2f}", fontsize=15)
plt.axis('off')  # Hide the axes
plt.show()
