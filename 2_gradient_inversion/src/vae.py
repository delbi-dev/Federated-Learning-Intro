# repository for the code used for my VAE model used to leverage priors in advance of producing a plausible structure from the aggregated gradient
# look at the notebook for a quick explanation and overview!

# to remove warnings on my pc, ignore it XD
import tensorflow as tf # type: ignore
import matplotlib.pyplot as plt # type: ignore

# LOADING MNIST
(x_train, _), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_train = x_train[..., None]
x_train = x_train[:10000]  # smaller subset of 10k as population

batch_size = 128
dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(10000).batch(batch_size)

# ENCODER
class Encoder(tf.keras.Model):
    def __init__(self, latent_dim=2):
        super().__init__()
        self.conv = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'),
            tf.keras.layers.Flatten()
        ])
        self.mu = tf.keras.layers.Dense(latent_dim)
        self.logvar = tf.keras.layers.Dense(latent_dim)
        
    def call(self, x):
        x = self.conv(x)
        return self.mu(x), self.logvar(x)
    
# DECODER
class Decoder(tf.keras.Model):
    def __init__(self, latent_dim=2):
        super().__init__()
        self.fc = tf.keras.layers.Dense(28*28*16, activation='relu')
        self.conv = tf.keras.Sequential([
            tf.keras.layers.Reshape((28,28,16)),
            tf.keras.layers.Conv2DTranspose(1, 3, activation='sigmoid', padding='same')
        ])
        
    def call(self, z):
        x = self.fc(z)
        return self.conv(x)

# VAE'S AT WORK
encoder = Encoder()
decoder = Decoder()
optimizer = tf.keras.optimizers.Adam(1e-3)

@tf.function
def train_step(x):
    with tf.GradientTape() as tape:
        mu, logvar = encoder(x)
        eps = tf.random.normal(shape=mu.shape)
        z = mu + tf.exp(0.5*logvar) * eps
        x_hat = decoder(z)
        recon_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(x, x_hat))
        kl_loss = -0.5 * tf.reduce_mean(1 + logvar - mu**2 - tf.exp(logvar))
        loss = recon_loss + kl_loss
    grads = tape.gradient(loss, encoder.trainable_variables + decoder.trainable_variables)
    optimizer.apply_gradients(zip(grads, encoder.trainable_variables + decoder.trainable_variables))
    return loss

viz_batch = next(iter(dataset))

for epoch in range(10): 
    for x in dataset:
        loss = train_step(x)

    # Visualization
    mu, logvar = encoder(viz_batch)
    eps = tf.random.normal(shape=mu.shape)
    z = mu + tf.exp(0.5 * logvar) * eps
    x_hat = decoder(z)

    plt.figure(figsize=(6, 3))

    # Reconstructed image
    plt.subplot(1, 2, 2)
    plt.imshow(x_hat[0, :, :, 0], cmap='gray')
    plt.title(f"Epoch {epoch+1}")
    plt.axis('off')

    plt.show()

    print(f"Epoch {epoch+1}, Loss: {loss.numpy():.4f}")