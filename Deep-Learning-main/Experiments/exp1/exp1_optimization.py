import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Set seeds for reproducibility
tf.random.set_seed(42)

def load_and_preprocess_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # Normalize pixel values
    x_train, x_test = x_train / 255.0, x_test / 255.0
    # Flatten the images
    x_train = x_train.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)
    # One-hot encode labels
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)

def build_model(initializer='glorot_uniform'):
    model = Sequential([
        Input(shape=(784,)),
        Dense(128, activation='relu', kernel_initializer=initializer),
        Dense(64, activation='relu', kernel_initializer=initializer),
        Dense(10, activation='softmax')
    ])
    return model

def plot_history(histories, title, ylabel='Accuracy'):
    plt.figure(figsize=(10, 6))
    for name, history in histories.items():
        val = history.history['accuracy'] if ylabel == 'Accuracy' else history.history['loss']
        plt.plot(val, label=name)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel('Epoch')
    plt.legend()
    # Save the plot
    filename = title.replace(' ', '_').lower() + '.png'
    plt.savefig(filename)
    print(f"Saved plot: {filename}")
    plt.close()

if __name__ == '__main__':
    print("Loading data...")
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    
    epochs = 10
    batch_size = 128
    
    # 1. Compare Initializations (using Adam)
    initializers = {
        'Random Normal': tf.keras.initializers.RandomNormal(mean=0., stddev=0.05),
        'Glorot Uniform (Xavier)': 'glorot_uniform',
        'He Normal': 'he_normal'
    }
    
    init_histories = {}
    print("--- Testing Initializations ---")
    for name, init in initializers.items():
        print(f"Training with {name} initialization...")
        model = build_model(initializer=init)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test), verbose=0)
        init_histories[name] = history
        print(f"Final Validation Accuracy ({name}): {history.history['val_accuracy'][-1]:.4f}")
        
    plot_history(init_histories, "Initialization Comparison - Accuracy", ylabel="Accuracy")
    plot_history(init_histories, "Initialization Comparison - Loss", ylabel="Loss")

    # 2. Compare Optimizers (using best init: He Normal)
    optimizers = {
        'SGD': 'sgd',
        'Adam': 'adam'
    }
    opt_histories = {}
    print("\n--- Testing Optimizers (with He Normal Init) ---")
    for name, opt in optimizers.items():
        print(f"Training with {name} optimizer...")
        model = build_model(initializer='he_normal')
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test), verbose=0)
        opt_histories[name] = history
        print(f"Final Validation Accuracy ({name}): {history.history['val_accuracy'][-1]:.4f}")

    plot_history(opt_histories, "Optimizer Comparison - Accuracy", ylabel="Accuracy")
    plot_history(opt_histories, "Optimizer Comparison - Loss", ylabel="Loss")
    
    print("\nExperiment 1 Complete.")
