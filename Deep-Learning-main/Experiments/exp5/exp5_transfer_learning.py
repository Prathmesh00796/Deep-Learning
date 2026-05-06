import os
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import cv2
import numpy as np

tf.random.set_seed(42)

def load_and_preprocess_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    # We will use a smaller subset to speed up training for the experiment
    x_train, y_train = x_train[:5000], y_train[:5000]
    x_test, y_test = x_test[:1000], y_test[:1000]
    
    # Preprocess inputs specifically for VGG16
    x_train = tf.keras.applications.vgg16.preprocess_input(x_train)
    x_test = tf.keras.applications.vgg16.preprocess_input(x_test)
    
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    return (x_train, y_train), (x_test, y_test)

def build_transfer_model():
    # Load VGG16 without the top classification layers
    # Input shape for CIFAR-10 is (32, 32, 3)
    base_model = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(32, 32, 3)))
    
    # Freeze the base layers
    for layer in base_model.layers:
        layer.trainable = False
        
    # Add custom classification head
    x = base_model.output
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(10, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    return model, base_model

def plot_history(history, title):
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()
    
    filename = title.replace(' ', '_').lower() + '.png'
    plt.savefig(filename)
    print(f"Saved plot: {filename}")
    plt.close()

if __name__ == '__main__':
    print("Loading data...")
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    
    print("Building model with frozen base...")
    model, base_model = build_transfer_model()
    
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    print("\n--- Phase 1: Training Classification Head ---")
    history_frozen = model.fit(x_train, y_train, epochs=5, batch_size=64, 
                               validation_data=(x_test, y_test), verbose=1)
    plot_history(history_frozen, "Phase 1 - Frozen Base Layers")
    
    print("\n--- Phase 2: Fine-Tuning ---")
    # Unfreeze the last few convolutional layers of VGG16
    for layer in base_model.layers[-4:]:
        layer.trainable = True
        
    # Recompile with a lower learning rate for fine-tuning
    model.compile(optimizer=Adam(learning_rate=1e-5), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    history_finetune = model.fit(x_train, y_train, epochs=5, batch_size=64, 
                                 validation_data=(x_test, y_test), verbose=1)
    plot_history(history_finetune, "Phase 2 - Fine Tuning")
    
    print("\nExperiment 5 Complete.")
