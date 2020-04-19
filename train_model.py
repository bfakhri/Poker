import tensorflow as tf
import numpy as np
import Dataset
import model
import datetime

# Params
batch_size = 2
training_steps = 100

# Get the dataset
ds = Dataset.CardDataset()

# Instantiate the model
model = model.CardClassifier()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

# Setup Tensorboard
log_dir = 'logs/' + datetime.datetime.now().strftime('%d-%H%M%S')
summary_writer = tf.summary.create_file_writer(log_dir, flush_millis=100)
summary_writer.set_as_default()


for step in range(training_steps):
    # Get batch of samples
    x, y = ds.batch_collection(batch_size)
    # Convert to float32
    x = tf.cast(x, tf.float32)/255.0
    y = tf.cast(y, tf.float32)

    with tf.GradientTape() as tape:
        y_hat = model(x)
        loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y, y_hat))

        grads = tape.gradient(loss, model.weights)
        optimizer.apply_gradients(zip(grads, model.weights))

        tf.summary.image('Inputs', x, step=step)
        tf.summary.scalar('ClassLoss', loss, step=step)
        tf.summary.histogram('Labels', y, step=step)
        tf.summary.histogram('Predictions', y_hat, step=step)

        print(step, loss)


