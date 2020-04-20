import tensorflow as tf
import numpy as np
import Dataset
import model
import datetime
import glob

# Hack to get it to work with RTX cards
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Params
batch_size = 64
training_steps = 1000000000
save_steps = 1000
models_dir = './saved_models/'
load_latest_model = True

def find_latest_model(model_dir):
    print('Searching for Saved Models in: ', model_dir)
    file_list = [f for f in glob.glob(model_dir+'*')]
    max_step = 0
    max_step_model = None
    for f in file_list:
        print('Evaluating ', f)
        step_str = f.split('_step_')[-1].split('/')[0]
        print('step_str: ', step_str)
        try:
            step = int(step_str)
            if(step > max_step):
                max_step = step
                max_step_model = f
        except: 
            continue

    print('Found latest model: ', max_step_model, max_step)
    return max_step_model, max_step

# Instantiate the model
if(load_latest_model):
    model_path, start_step = find_latest_model(models_dir)
    model = model = tf.keras.models.load_model(model_path)
else:
    model = model.CardClassifier()
    model._set_inputs(tf.keras.Input(shape=(None, None, 3)))
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

# Get the dataset
ds = Dataset.CardDataset()

# Setup Tensorboard
log_dir = 'logs/' + datetime.datetime.now().strftime('%d-%H%M%S')
summary_writer = tf.summary.create_file_writer(log_dir, flush_millis=100)
summary_writer.set_as_default()

def accuracy(y, y_hat, thresh=0.5):
    y_hat_bool = y_hat > thresh
    y_bool = y > thresh
    acc = tf.reduce_mean(tf.cast(tf.equal(y_bool, y_hat_bool), tf.float32))
    return acc

def precision_recall(y, y_hat, thresh=0.5):
    y_hat_bool = y_hat > thresh
    y_bool = y > thresh
    TP = tf.reduce_sum(tf.cast(tf.logical_and(y_bool, y_hat_bool), tf.float32))
    FP = tf.reduce_sum(tf.cast(tf.logical_and(y_bool, tf.logical_not(y_hat_bool)), tf.float32))
    precision = TP/(TP+FP)
    recall = TP/tf.reduce_sum(tf.cast(y_bool, tf.float32))
    return precision, recall

for step in range(start_step, training_steps):
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

        acc = accuracy(y, y_hat)
        precision, recall = precision_recall(y, y_hat)
        tf.summary.image('Inputs', x, step=step)
        tf.summary.scalar('ClassLoss', loss, step=step)
        tf.summary.scalar('Acc', acc, step=step)
        tf.summary.scalar('Precision', precision, step=step)
        tf.summary.scalar('Recall', recall, step=step)
        tf.summary.histogram('Labels', y, step=step)
        tf.summary.histogram('Predictions', y_hat, step=step)

        print('Step: ', step, acc.numpy()*100, precision.numpy(), recall.numpy(), loss.numpy())

    # Save model
    if(step % save_steps == 0):
        model_path = models_dir + model.name + '_step_' + str(step) + '/'
        model.save(model_path)


