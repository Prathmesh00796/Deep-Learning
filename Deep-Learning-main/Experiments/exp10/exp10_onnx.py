import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

try:
    import tf2onnx
    import onnxruntime as ort
except ImportError:
    print("Please install tf2onnx and onnxruntime:")
    print("pip install tf2onnx onnxruntime")
    exit()

tf.random.set_seed(42)

def build_and_train_model():
    print("Loading data...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    print("Building model...")
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    print("Training model...")
    model.fit(x_train, y_train, epochs=1, batch_size=64, validation_split=0.1, verbose=1)
    
    return model, x_test, y_test

if __name__ == '__main__':
    model, x_test, y_test = build_and_train_model()
    
    # Save the model to SavedModel format first
    saved_model_path = "saved_model"
    model.save(saved_model_path)
    print(f"Model saved to {saved_model_path}")
    
    # Convert to ONNX
    onnx_model_path = "model.onnx"
    print("Converting model to ONNX format...")
    # Note: tf2onnx can convert directly from tf.keras models in memory
    spec = (tf.TensorSpec((None, 28, 28), tf.float32, name="input"),)
    output_path = onnx_model_path
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=output_path)
    print(f"ONNX model saved to {onnx_model_path}")
    
    # Inference using ONNX Runtime
    print("\nLoading ONNX model for inference...")
    session = ort.InferenceSession(onnx_model_path)
    
    # Get the input name for the ONNX session
    input_name = session.get_inputs()[0].name
    
    # Prepare input data (using a single test image)
    test_image = x_test[0:1].astype(np.float32)
    true_label = y_test[0]
    
    print("Running inference...")
    result = session.run(None, {input_name: test_image})
    
    # Result is a list of outputs; we take the first output (softmax probabilities)
    predicted_probs = result[0][0]
    predicted_label = np.argmax(predicted_probs)
    
    print(f"True Label: {true_label}")
    print(f"Predicted Label (ONNX Runtime): {predicted_label}")
    
    if true_label == predicted_label:
        print("ONNX Inference successful and matches true label!")
    else:
        print("ONNX Inference failed to match true label.")
        
    print("\nExperiment 10 Complete.")
