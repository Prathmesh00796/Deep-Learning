import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

tf.random.set_seed(42)

def load_and_preprocess_data():
    # Use fewer samples to deliberately cause overfitting for demonstration purposes
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    # Subsample training data to 5000 samples to encourage overfitting
    return (x_train[:5000], y_train[:5000]), (x_test, y_test)

def build_model(config='baseline'):
    model = Sequential()
    model.add(Input(shape=(784,)))
    
    if config == 'baseline':
        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
    
    elif config == 'dropout':
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        
    elif config == 'l2':
        model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.01)))
        model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))
        
    elif config == 'batch_norm':
        model.add(Dense(256, use_bias=False))
        model.add(BatchNormalization())
        model.add(tf.keras.layers.Activation('relu'))
        
        model.add(Dense(128, use_bias=False))
        model.add(BatchNormalization())
        model.add(tf.keras.layers.Activation('relu'))

    model.add(Dense(10, activation='softmax'))
    return model

def plot_history(histories, title):
    plt.figure(figsize=(12, 8))
    for name, history in histories.items():
        val = history.history['val_loss']
        train = history.history['loss']
        epochs = range(1, len(val) + 1)
        plt.plot(epochs, train, label=f"{name} Train")
        plt.plot(epochs, val, '--', label=f"{name} Val")
        
    plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    filename = title.replace(' ', '_').lower() + '.png'
    plt.savefig(filename)
    print(f"Saved plot: {filename}")
    plt.close()

if __name__ == '__main__':
    print("Loading data...")
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    
    epochs = 20
    batch_size = 64
    
    configs = ['baseline', 'dropout', 'l2', 'batch_norm']
    histories = {}
    
    for config in configs:
        print(f"\nTraining model with config: {config}")
        model = build_model(config)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, 
                            validation_data=(x_test, y_test), verbose=0)
        histories[config] = history
        print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
        print(f"Final Validation Loss: {history.history['val_loss'][-1]:.4f}")
        print(f"Final Training Loss: {history.history['loss'][-1]:.4f}")
        
    plot_history(histories, "Regularization Techniques Comparison - Loss")
    print("\nExperiment 2 Complete.")
