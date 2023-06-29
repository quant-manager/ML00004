#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2017 François Chollet
# IGNORE_COPYRIGHT: cleared by OSS licensing
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
# Source (article):
# https://www.tensorflow.org/tutorials/images/transfer_learning
# Source (Jupyter notebook):
# https://storage.googleapis.com/tensorflow_docs/docs/site/en/tutorials/images/transfer_learning.ipynb
# Source (GitHub):
# https://github.com/tensorflow/docs/blob/master/site/en/tutorials/images/transfer_learning.ipynb
# Source (Google Colab):
# https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/images/transfer_learning.ipynb?force_kitty_mode=1&force_corgi_mode=1

"""
@author: James J Johnson
@url: https://www.linkedin.com/in/james-james-johnson/
"""

###############################################################################
# Import packages

import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

###############################################################################
# Report CPU/GPU availability.
print("Tensorflow version: {version}".format(version=tf.__version__))
print()
print("Fitting will be using {int_cpu_count:d} CPU(s).".format(
    int_cpu_count = len(tf.config.list_physical_devices('CPU'))))
print("Fitting will be using {int_gpu_count:d} GPU(s).".format(
    int_gpu_count = len(tf.config.list_physical_devices('GPU'))))
print()

###############################################################################
# Download and preprocess images data

_URL='https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
path_to_zip = tf.keras.utils.get_file(
    'cats_and_dogs.zip',
    origin=_URL, extract=True)
PATH = os.path.join(
    os.path.dirname(path_to_zip),
    'cats_and_dogs_filtered')

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

BATCH_SIZE = 32
IMG_SIZE = (160, 160)

train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    shuffle=True,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    validation_dir,
    shuffle=True,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE)

###############################################################################
# Show a few sample images

class_names = train_dataset.class_names
plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
plt.savefig("sample_images.pdf",
            format="pdf", bbox_inches="tight")

###############################################################################
# Create test set

# As the original dataset doesn't contain a test set, you will create one.
# To do so, determine how many batches of data are available in the validation
# set using tf.data.experimental.cardinality, then move 20% of them to a
# test set.

val_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(val_batches // 5)
validation_dataset = validation_dataset.skip(val_batches // 5)
print('Number of validation batches: %d' % tf.data.experimental.cardinality(
    validation_dataset))
print('Number of test batches: %d' % tf.data.experimental.cardinality(
    test_dataset))

###############################################################################
# Configure the dataset for performance
# Use buffered prefetching to load images from disk without having I/O become
# blocking: https://www.tensorflow.org/guide/data_performance

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

###############################################################################
# Use data augmentation

# These layers are active only during training, when you call Model.fit. They
# are inactive when the model is used in inference mode in Model.evaluate or
# Model.predict.

data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip('horizontal'),
  tf.keras.layers.RandomRotation(0.2),
])

###############################################################################
# Repeatedly apply these augmentation layers to the same image to show the
# impact.

for image, _ in train_dataset.take(1):
  plt.figure(figsize=(10, 10))
  first_image = image[0]
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
    plt.imshow(augmented_image[0] / 255)
    plt.axis('off')
plt.savefig("augmented_sample_image.pdf",
            format="pdf", bbox_inches="tight")

###############################################################################
# Rescale pixel values

# In a moment, you will download tf.keras.applications.MobileNetV2 for use as
# your base model. This model expects pixel values in [-1, 1], but at this
# point, the pixel values in your images are in [0, 255]. To rescale them,
# use the preprocessing method included with the model

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
# Alternatively, you could rescale pixel values from [0, 255] to [-1, 1]
# using tf.keras.layers.Rescaling.
rescale = tf.keras.layers.Rescaling(1./127.5, offset=-1)
# If using other tf.keras.applications, be sure to check the API doc to
# determine if they expect pixels in [-1, 1] or [0, 1], or use the included
# preprocess_input function.

###############################################################################
# Create the base model from the pre-trained convnets

# Create the base model from the MobileNet V2 model developed at Google.
# This is pre-trained on the ImageNet dataset, a large dataset consisting of
# 1.4M images and 1000 classes.

# Create the base model from the pre-trained model MobileNet V2
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SHAPE,
    include_top=False,
    weights='imagenet')

image_batch, label_batch = next(iter(train_dataset))
feature_batch = base_model(image_batch)
print(feature_batch.shape)

###############################################################################
# Feature extraction

base_model.trainable = False # Freeze the convolutional base: all layers

# Important note about BatchNormalization layers

# Many models contain tf.keras.layers.BatchNormalization layers. This layer is
# a special case and precautions should be taken in the context of fine-tuning.
# When you set layer.trainable = False, the BatchNormalization layer will run
# in inference mode, and will not update its mean and variance statistics.
# When you unfreeze a model that contains BatchNormalization layers in order to
# do fine-tuning, you should keep the BatchNormalization layers in inference
# mode by passing training = False when calling the base model. Otherwise, the
# updates applied to the non-trainable weights will destroy what the model has
# learned.

base_model.summary()

###############################################################################
# Create and add a classification head

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
prediction_layer = tf.keras.layers.Dense(1)
prediction_batch = prediction_layer(feature_batch_average)

print(feature_batch_average.shape)
print(prediction_batch.shape)

inputs = tf.keras.Input(shape=(160, 160, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

###############################################################################
# Compile the model
base_learning_rate = 0.0001
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy'])
model.summary()
# The 2.5 million parameters in MobileNet are frozen, but there are 1.2
# thousand trainable parameters in the Dense layer. These are divided between
# two tf.Variable objects, the weights and biases.
print(len(model.trainable_variables))

###############################################################################
# Train the model

initial_epochs = 10

loss0, accuracy0 = model.evaluate(validation_dataset)
print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))
history = model.fit(
    train_dataset,
    epochs=initial_epochs,
    validation_data=validation_dataset)
loss1, accuracy1 = model.evaluate(validation_dataset)
print("final loss: {:.2f}".format(loss1))
print("final accuracy: {:.2f}".format(accuracy1))

###############################################################################
# Plot learning curves

# If you are wondering why the validation metrics are clearly better than the
# training metrics, the main factor is because layers like
# tf.keras.layers.BatchNormalization and tf.keras.layers.Dropout affect
# accuracy during training. They are turned off when calculating validation
# loss.

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

###############################################################################
# Fine tuning

# In the feature extraction experiment, you were only training a few layers on
# top of an MobileNetV2 base model. The weights of the pre-trained network were
# not updated during training.

# One way to increase performance even further is to train (or "fine-tune")
# the weights of the top layers of the pre-trained model alongside the training
# of the classifier you added. The training process will force the weights to
# be tuned from generic feature maps to features associated specifically with
# the dataset.

# Note: This should only be attempted after you have trained the top-level
# classifier with the pre-trained model set to non-trainable. If you add a
# randomly initialized classifier on top of a pre-trained model and attempt to
# train all layers jointly, the magnitude of the gradient updates will be too
# large (due to the random weights from the classifier) and your pre-trained
# model will forget what it has learned.

# Also, you should try to fine-tune a small number of top layers rather than
# the whole MobileNet model. In most convolutional networks, the higher up a
# layer is, the more specialized it is. The first few layers learn very simple
# and generic features that generalize to almost all types of images. As you go
# higher up, the features are increasingly more specific to the dataset on
# which the model was trained. The goal of fine-tuning is to adapt these
# specialized features to work with the new dataset, rather than overwrite
# the generic learning.

###############################################################################

# Un-freeze the top layers of the model

base_model.trainable = True

# Let's take a look to see how many layers are in the base model
print("Number of layers in the base model: ", len(base_model.layers))

# Fine-tune from this layer onwards
fine_tune_at = 100

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable = False

###############################################################################
# Compile the model

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate/10),
    metrics=['accuracy'])

model.summary()
print(len(model.trainable_variables))

# Continue training the model

###############################################################################
# If you trained to convergence earlier, this step will improve your accuracy
# by a few percentage points.

fine_tune_epochs = 10
total_epochs =  initial_epochs + fine_tune_epochs

history_fine = model.fit(
    train_dataset,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1],
    validation_data=validation_dataset)

###############################################################################
# Plot learning curves

# Let's take a look at the learning curves of the training and validation
# accuracy/loss when fine-tuning the last few layers of the MobileNetV2 base
# model and training the classifier on top of it. The validation loss is much
# higher than the training loss, so you may get some overfitting.

# You may also get some overfitting as the new training set is relatively small
# and similar to the original MobileNetV2 datasets.

# After fine tuning the model nearly reaches 98% accuracy on the validation set.

acc += history_fine.history['accuracy']
val_acc += history_fine.history['val_accuracy']

loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0.8, 1])
plt.plot([initial_epochs-1,initial_epochs-1],
          plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 1.0])
plt.plot([initial_epochs-1,initial_epochs-1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

plt.savefig("learning_curves_fine_tuned.pdf",
            format="pdf", bbox_inches="tight")

###############################################################################
# Evaluation and prediction

loss, accuracy = model.evaluate(test_dataset)
print('Test accuracy :', accuracy)

# Retrieve a batch of images from the test set
image_batch, label_batch = test_dataset.as_numpy_iterator().next()
predictions = model.predict_on_batch(image_batch).flatten()

# Apply a sigmoid since our model returns logits
predictions = tf.nn.sigmoid(predictions)
predictions = tf.where(predictions < 0.5, 0, 1)

print('Predictions:\n', predictions.numpy())
print('Labels:\n', label_batch)

plt.figure(figsize=(10, 10))
for i in range(9):
  ax = plt.subplot(3, 3, i + 1)
  plt.imshow(image_batch[i].astype("uint8"))
  plt.title(class_names[predictions[i]])
  plt.axis("off")

plt.savefig("test_set_sample_predictions.pdf",
            format="pdf", bbox_inches="tight")

###############################################################################
# Summary

# Using a pre-trained model for feature extraction:
# When working with a small dataset, it is a common practice to take advantage
# of features learned by a model trained on a larger dataset in the same
# domain. This is done by instantiating the pre-trained model and adding a
# fully-connected classifier on top. The pre-trained model is "frozen" and only
# the weights of the classifier get updated during training. In this case, the
# convolutional base# extracted all the features associated with each image and
# you just trained a classifier that determines the image class given that
# set of extracted features.

# Fine-tuning a pre-trained model:
# To further improve performance, one might want to repurpose the top-level
# layers of the pre-trained models to the new dataset via fine-tuning. In this
# case, you tuned your weights such that your model learned high-level features
# specific to the dataset. This technique is usually recommended when the
# training dataset is large and very similar to the original dataset that the
# pre-trained model was trained on.
