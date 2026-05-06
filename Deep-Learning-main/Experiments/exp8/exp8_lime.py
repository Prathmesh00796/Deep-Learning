import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

try:
    from lime import lime_image
    from skimage.segmentation import mark_boundaries
except ImportError:
    print("Please install LIME and scikit-image:")
    print("pip install lime scikit-image")
    exit()

tf.random.set_seed(42)

def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # LIME Image Explainer expects RGB images usually, so we'll convert grayscale to RGB by repeating channels
    x_train = np.stack((x_train,) * 3, axis=-1) / 255.0
    x_test = np.stack((x_test,) * 3, axis=-1) / 255.0
    
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)

def build_cnn():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    return model

if __name__ == '__main__':
    print("Loading data...")
    (x_train, y_train), (x_test, y_test) = load_data()
    
    print("Building and training CNN...")
    model = build_cnn()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # Train for just 2 epochs to save time
    model.fit(x_train, y_train, epochs=2, batch_size=64, validation_split=0.1, verbose=1)
    
    print("\nInitializing LIME Explainer...")
    explainer = lime_image.LimeImageExplainer()
    
    # Pick a random image from the test set
    image_idx = 0
    image = x_test[image_idx]
    true_label = np.argmax(y_test[image_idx])
    
    # Model's predict function wrapper for LIME
    def predict_fn(images):
        return model.predict(images, verbose=0)
    
    print(f"Generating explanation for image {image_idx} (True Label: {true_label})...")
    explanation = explainer.explain_instance(image, predict_fn, top_labels=1, hide_color=0, num_samples=1000)
    
    # Get the top predicted label
    top_label = explanation.top_labels[0]
    
    # Get the image and mask for the top label
    temp, mask = explanation.get_image_and_mask(top_label, positive_only=True, num_features=5, hide_rest=False)
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.title(f"Original Image (True: {true_label}, Pred: {top_label})")
    plt.imshow(image)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("LIME Explanation (Green regions contributed to prediction)")
    plt.imshow(mark_boundaries(temp, mask))
    plt.axis('off')
    
    plt.savefig('lime_explanation.png')
    print("Saved explanation to lime_explanation.png")
    
    print("\nExperiment 8 Complete.")
