import tensorflow as tf
import numpy as np
import Dataset
import model
import datetime
import glob
from tensorflow.python.keras import backend as K
import utils

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
pos_weight = 10.0
neg_weight = 1.0/pos_weight

# Instantiate the model
latest_model_path, start_step = utils.find_latest_model(models_dir)
if(load_latest_model and latest_model_path is not None):
    model = model = tf.keras.models.load_model(latest_model_path)
else:
    start_step = 0
    model = model.CardClassifier()
    model._set_inputs(tf.keras.Input(shape=(None, None, 3)))
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

# Get the dataset
ds = Dataset.CardDataset()

# Setup Tensorboard
log_dir = 'logs/' + datetime.datetime.now().strftime('%d-%H%M%S')
summary_writer = tf.summary.create_file_writer(log_dir, flush_millis=100)
summary_writer.set_as_default()


for step in range(start_step, training_steps):
    # Get batch of samples
    x, y = ds.batch_collection(batch_size)
    # Convert to float32
    x = tf.cast(x, tf.float32)/255.0
    y = tf.cast(y, tf.float32)

    with tf.GradientTape() as tape:
        y_hat = model(x)
        loss = K.binary_crossentropy(y, y_hat)

        # Weight the loss
        pos_weights = y*pos_weight
        neg_weights = (1.0-y)*neg_weight
        loss_weights = pos_weights + neg_weights

        loss_weighted = tf.reduce_mean(loss*loss_weights)

        grads = tape.gradient(loss_weighted, model.weights)
        optimizer.apply_gradients(zip(grads, model.weights))

        acc = utils.accuracy(y, y_hat)
        acc_topk = utils.accuracy_topk(y, y_hat)
        precision, recall = utils.precision_recall(y, y_hat)
        auc_metric = tf.keras.metrics.AUC()
        auc_metric.update_state(y, y_hat)
        tf.summary.image('Inputs', x, step=step)
        tf.summary.scalar('ClassLoss', loss_weighted, step=step)
        tf.summary.scalar('Acc', acc, step=step)
        tf.summary.scalar('AuC', auc_metric.result(), step=step)
        tf.summary.scalar('AccTopK', acc_topk, step=step)
        tf.summary.scalar('Precision', precision, step=step)
        tf.summary.scalar('Recall', recall, step=step)
        tf.summary.histogram('Labels', y, step=step)
        tf.summary.histogram('Predictions', y_hat, step=step)
        auc_metric.reset_states()

        print('Step: ', step, acc.numpy()*100, precision.numpy(), recall.numpy(), loss_weighted.numpy())

    # Save model
    if(step % save_steps == 0):
        model_path = models_dir + model.name + '_step_' + str(step) + '/'
        model.save(model_path)


