import tensorflow as tf
import numpy as np
import Dataset
import model
import datetime
# Hack to get it to work with RTX cards
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Params
batch_size = 64
training_steps = 1000000000
save_steps = 1000
models_dir = './saved_models/'

# Get the dataset
ds = Dataset.CardDataset()

# Instantiate the model
model = model.CardClassifier()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

# Setup Tensorboard
log_dir = 'logs/' + datetime.datetime.now().strftime('%d-%H%M%S')
summary_writer = tf.summary.create_file_writer(log_dir, flush_millis=100)
summary_writer.set_as_default()

def accuracy(y, y_hat, thresh=0.5):
    y_hat_bool = y_hat > thresh
    y_bool = y > thresh
    acc = tf.reduce_mean(tf.cast(tf.equal(y_bool, y_hat_bool), tf.float32))
    return acc

for step in range(training_steps):
    # Get batch of samples
    x, y = ds.batch_collection(batch_size)
    # Convert to float32
    x = tf.cast(x, tf.float32)/255.0
    y = tf.cast(y, tf.float32)

    if(step == 0):
        model._set_inputs(x)

    with tf.GradientTape() as tape:
        y_hat = model(x)
        loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y, y_hat))

        grads = tape.gradient(loss, model.weights)
        optimizer.apply_gradients(zip(grads, model.weights))

        acc = accuracy(y, y_hat)
        tf.summary.image('Inputs', x, step=step)
        tf.summary.scalar('ClassLoss', loss, step=step)
        tf.summary.scalar('Acc', acc, step=step)
        tf.summary.histogram('Labels', y, step=step)
        tf.summary.histogram('Predictions', y_hat, step=step)

        print(step, acc.numpy()*100, loss.numpy())

    # Save model
    if(step % save_steps == 0):
        model_path = models_dir + model.name + '/'
        model.save(model_path)


