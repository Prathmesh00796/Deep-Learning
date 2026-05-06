import os
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, GRU, Dense
import matplotlib.pyplot as plt

tf.random.set_seed(42)

def load_and_preprocess_data(max_features=10000, maxlen=200):
    print("Loading IMDb data...")
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    
    # Pad sequences so they all have the same length
    x_train = pad_sequences(x_train, maxlen=maxlen)
    x_test = pad_sequences(x_test, maxlen=maxlen)
    
    return (x_train, y_train), (x_test, y_test)

def build_model(model_type, max_features=10000, maxlen=200):
    model = Sequential()
    model.add(Embedding(input_dim=max_features, output_dim=32, input_length=maxlen))
    
    if model_type == 'RNN':
        model.add(SimpleRNN(32))
    elif model_type == 'LSTM':
        model.add(LSTM(32))
    elif model_type == 'GRU':
        model.add(GRU(32))
        
    model.add(Dense(1, activation='sigmoid'))
    return model

def plot_history(histories, title):
    plt.figure(figsize=(12, 6))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    for name, history in histories.items():
        plt.plot(history.history['val_accuracy'], label=name)
    plt.title(f"{title} - Validation Accuracy")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    for name, history in histories.items():
        plt.plot(history.history['val_loss'], label=name)
    plt.title(f"{title} - Validation Loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    filename = title.replace(' ', '_').lower() + '.png'
    plt.savefig(filename)
    print(f"Saved plot: {filename}")
    plt.close()

if __name__ == '__main__':
    max_features = 10000
    maxlen = 100 # Keep it short for faster training in experiment
    
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data(max_features, maxlen)
    
    epochs = 5
    batch_size = 128
    
    models_to_test = ['RNN', 'LSTM', 'GRU']
    histories = {}
    
    for m_type in models_to_test:
        print(f"\nTraining {m_type} Model...")
        model = build_model(m_type, max_features, maxlen)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, 
                            validation_data=(x_test, y_test), verbose=1)
        histories[m_type] = history
        
        loss, acc = model.evaluate(x_test, y_test, verbose=0)
        print(f"Final Test Accuracy ({m_type}): {acc:.4f}")
        
    plot_history(histories, "Sequential Models Comparison")
    
    print("\nExperiment 6 Complete.")
