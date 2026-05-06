import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

tf.random.set_seed(42)

def load_and_preprocess_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    # Normalize pixel values
    x_train, x_test = x_train / 255.0, x_test / 255.0
    # One-hot encode labels
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)

def build_cnn():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3), name='conv1'),
        MaxPooling2D((2, 2), name='pool1'),
        Conv2D(64, (3, 3), activation='relu', name='conv2'),
        MaxPooling2D((2, 2), name='pool2'),
        Conv2D(64, (3, 3), activation='relu', name='conv3'),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    return model

def visualize_feature_maps(model, image):
    # Extract outputs of the first convolutional layer
    layer_outputs = [layer.output for layer in model.layers if 'conv' in layer.name or 'pool' in layer.name]
    activation_model = Model(inputs=model.input, outputs=layer_outputs)
    
    # Get activations for the input image
    img_tensor = np.expand_dims(image, axis=0)
    activations = activation_model.predict(img_tensor, verbose=0)
    
    # Plot feature maps of the first layer (conv1)
    first_layer_activation = activations[0]
    
    plt.figure(figsize=(10, 10))
    # Plot up to 16 feature maps
    for i in range(min(16, first_layer_activation.shape[-1])):
        ax = plt.subplot(4, 4, i + 1)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(first_layer_activation[0, :, :, i], cmap='viridis')
    
    plt.suptitle("Feature Maps of First Conv Layer")
    plt.savefig('feature_maps.png')
    print("Saved feature maps to feature_maps.png")
    plt.close()

if __name__ == '__main__':
    print("Loading CIFAR-10 data...")
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    
    print("Building CNN Model...")
    model = build_cnn()
    model.summary()
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    epochs = 10
    batch_size = 64
    
    print("Training CNN...")
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, 
                        validation_data=(x_test, y_test), verbose=1)
    
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nFinal Test Accuracy: {accuracy:.4f}")
    
    print("Visualizing Feature Maps...")
    # Use the first test image
    sample_image = x_test[0]
    visualize_feature_maps(model, sample_image)
    
    print("\nExperiment 4 Complete.")
