import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

tf.random.set_seed(42)

def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)

def build_model():
    model = Sequential([
        Input(shape=(784,)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    return model

def step_decay(epoch, lr):
    # Drop learning rate by a factor of 0.5 every 5 epochs
    drop = 0.5
    epochs_drop = 5.0
    if (epoch + 1) % epochs_drop == 0:
        return lr * drop
    return lr

def plot_history(histories, title):
    plt.figure(figsize=(10, 6))
    for name, history in histories.items():
        plt.plot(history.history['val_loss'], label=f"{name} Val Loss")
        
    plt.title(title)
    plt.ylabel('Validation Loss')
    plt.xlabel('Epoch')
    plt.legend()
    filename = title.replace(' ', '_').lower() + '.png'
    plt.savefig(filename)
    print(f"Saved plot: {filename}")
    plt.close()

if __name__ == '__main__':
    print("Loading data...")
    (x_train, y_train), (x_test, y_test) = load_data()
    
    epochs = 30
    batch_size = 128
    
    histories = {}
    
    # 1. Fixed Learning Rate (No callbacks)
    print("\nTraining with Fixed Learning Rate...")
    model_fixed = build_model()
    model_fixed.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1), 
                        loss='categorical_crossentropy', metrics=['accuracy'])
    hist_fixed = model_fixed.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, 
                                 validation_data=(x_test, y_test), verbose=0)
    histories['Fixed LR'] = hist_fixed
    
    # 2. Learning Rate Scheduling (Step Decay)
    print("Training with Learning Rate Scheduling...")
    model_decay = build_model()
    model_decay.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1), 
                        loss='categorical_crossentropy', metrics=['accuracy'])
    lr_scheduler = LearningRateScheduler(step_decay, verbose=0)
    hist_decay = model_decay.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, 
                                 validation_data=(x_test, y_test), callbacks=[lr_scheduler], verbose=0)
    histories['LR Decay'] = hist_decay
    
    # 3. Early Stopping
    print("Training with Early Stopping...")
    model_es = build_model()
    model_es.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1), 
                     loss='categorical_crossentropy', metrics=['accuracy'])
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)
    hist_es = model_es.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, 
                           validation_data=(x_test, y_test), callbacks=[early_stop], verbose=0)
    histories['Early Stopping'] = hist_es
    
    plot_history(histories, "Learning Rate Scheduling and Early Stopping Comparison")
    
    print("\nExperiment 3 Complete.")
