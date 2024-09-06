import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. Data Collection & Preprocessing

# Specify the directory containing your image data
data_dir = 'Dataset/glioma'  # Replace with the actual path to your image folder
# Set image dimensions and batch size
img_height, img_width = 1280, 1024  # Adjust as needed
batch_size = 32

# Create ImageDataGenerator for data loading and preprocessing
datagen = ImageDataGenerator(rescale=1./255)  # Normalize pixel values

# Load images from the directory
train_data_gen = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='input',  # Unsupervised learning
    color_mode='rgb',  # Or 'grayscale' if applicable
    shuffle=True
)
print(f"Found {train_data_gen.samples} images.")

# 2. Model Selection & Training (GAN for Images - using Convolutions)

def make_generator_model():
    model = Sequential()
    model.add(layers.Dense(16384, use_bias=False, input_shape=(16384,)))
    inputs = layers.Input(shape=(100,))
    x = layers.Dense(8 * 8 * 256, use_bias=False)(inputs)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())


    model.add(layers.Reshape((8, 8, 256)))

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    return model

def make_discriminator_model():
    model = Sequential()
    model.add(layers.Input(shape=(2660, 2180, 3)))  # Match your image dimensions and channels
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
    # ... rest of your discriminator model
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))

    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    return model

# Loss functions
cross_entropy = BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# Optimizers
generator_optimizer = Adam(1e-4)
discriminator_optimizer = Adam(1e-4)

generator = make_generator_model()
discriminator = make_discriminator_model()
for image_batch, _ in train_data_gen:
    if image_batch.shape[0] == 0:
        print("Empty batch detected. Skipping this batch.")
        break
    print(f"Image batch shape: {image_batch.shape}")
    # Proceed with training steps

# Training loop
EPOCHS = 50
noise_dim = 100
for epoch in range(EPOCHS):
    for image_batch in train_data_gen:
        # Train the discriminator
        noise = tf.random.normal([batch_size, 16384])
        with tf.GradientTape() as disc_tape:
            generated_images = generator(noise, training=True)
            real_output = discriminator(image_batch, training=True)
            fake_output = discriminator(generated_images, training=True)

            disc_loss = discriminator_loss(real_output, fake_output)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

        # Train the generator 

        noise = tf.random.normal([batch_size, 16384])
        with tf.GradientTape() as gen_tape:
            generated_images = generator(noise, training=True)
            fake_output = discriminator(generated_images, training=True)
            gen_loss = generator_loss(fake_output)
        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables) 

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
# 3. Synthetic Data Generation & Evaluation
# Generate a batch of synthetic images
noise = tf.random.normal([batch_size, 16384])
generated_images = generator(noise, training=False)
save_dir = 'Synthetic Data'
os.makedirs(save_dir, exist_ok=True)
for i in range(synthetic_images.shape[0]):
    filename = os.path.join(save_dir, f'synthetic_image_{i}.png')  # Or use another image format like .jpg
    plt.imsave(filename, synthetic_images[i, :, :, :])

# Visualize generated images
plt.figure(figsize=(10,10))
for i in range(generated_images.shape[0]):
    plt.subplot(4, 4, i+1)
    plt.imshow(generated_images[i, :, :, :])
    plt.axis('off')
plt.show()