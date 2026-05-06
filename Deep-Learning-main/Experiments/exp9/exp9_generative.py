import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape, Input, Lambda, LeakyReLU
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

tf.random.set_seed(42)

def load_data():
    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    return x_train

# --- VAE Implementation ---
def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def build_vae(input_dim=784, latent_dim=2):
    # Encoder
    inputs = Input(shape=(input_dim,), name='encoder_input')
    x = Dense(256, activation='relu')(inputs)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)
    
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    
    # Decoder
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(256, activation='relu')(latent_inputs)
    outputs = Dense(input_dim, activation='sigmoid')(x)
    decoder = Model(latent_inputs, outputs, name='decoder')
    
    # VAE
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae_mlp')
    
    # Loss
    reconstruction_loss = binary_crossentropy(inputs, outputs)
    reconstruction_loss *= input_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    
    return vae, decoder

# --- GAN Implementation ---
def build_gan(latent_dim=100, input_dim=784):
    # Generator
    generator = Sequential([
        Dense(256, input_dim=latent_dim),
        LeakyReLU(alpha=0.2),
        Dense(512),
        LeakyReLU(alpha=0.2),
        Dense(input_dim, activation='sigmoid')
    ], name="generator")
    
    # Discriminator
    discriminator = Sequential([
        Dense(512, input_dim=input_dim),
        LeakyReLU(alpha=0.2),
        Dense(256),
        LeakyReLU(alpha=0.2),
        Dense(1, activation='sigmoid')
    ], name="discriminator")
    
    discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    discriminator.trainable = False
    
    # GAN
    gan_input = Input(shape=(latent_dim,))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(gan_input, gan_output, name="gan")
    gan.compile(loss='binary_crossentropy', optimizer='adam')
    
    return generator, discriminator, gan

def plot_generated_images(generator, decoder, epoch, latent_dim_gan=100, latent_dim_vae=2):
    plt.figure(figsize=(10, 5))
    
    # GAN Images
    noise = np.random.normal(0, 1, (5, latent_dim_gan))
    gen_imgs_gan = generator.predict(noise, verbose=0)
    
    for i in range(5):
        plt.subplot(2, 5, i + 1)
        plt.imshow(gen_imgs_gan[i].reshape(28, 28), cmap='gray')
        plt.title("GAN")
        plt.axis('off')
        
    # VAE Images
    z_sample = np.random.normal(0, 1, (5, latent_dim_vae))
    gen_imgs_vae = decoder.predict(z_sample, verbose=0)
    
    for i in range(5):
        plt.subplot(2, 5, i + 6)
        plt.imshow(gen_imgs_vae[i].reshape(28, 28), cmap='gray')
        plt.title("VAE")
        plt.axis('off')
        
    plt.tight_layout()
    plt.savefig(f"generative_comparison_epoch_{epoch}.png")
    plt.close()

if __name__ == '__main__':
    print("Loading data...")
    x_train = load_data()
    
    print("--- Training VAE ---")
    vae, vae_decoder = build_vae(latent_dim=2)
    vae.compile(optimizer='adam')
    # Train VAE for 5 epochs
    vae.fit(x_train, epochs=5, batch_size=128, verbose=1)
    
    print("\n--- Training GAN ---")
    latent_dim_gan = 100
    generator, discriminator, gan = build_gan(latent_dim=latent_dim_gan)
    
    epochs = 1000 # GAN needs more iterations, but keeping it low for experiment demo
    batch_size = 128
    half_batch = int(batch_size / 2)
    
    for epoch in range(epochs):
        # Train Discriminator
        idx = np.random.randint(0, x_train.shape[0], half_batch)
        imgs = x_train[idx]
        noise = np.random.normal(0, 1, (half_batch, latent_dim_gan))
        gen_imgs = generator.predict(noise, verbose=0)
        
        d_loss_real = discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # Train Generator
        noise = np.random.normal(0, 1, (batch_size, latent_dim_gan))
        valid_y = np.array([1] * batch_size)
        g_loss = gan.train_on_batch(noise, valid_y)
        
        if epoch % 500 == 0:
            print(f"GAN Epoch {epoch} [D loss: {d_loss[0]:.4f}, acc.: {100*d_loss[1]:.2f}%] [G loss: {g_loss:.4f}]")
    
    print("Generating Comparison Images...")
    plot_generated_images(generator, vae_decoder, epoch=epochs)
    print("Saved comparison image.")
    print("\nExperiment 9 Complete.")
