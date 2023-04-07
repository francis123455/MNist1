import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from model import build_model, build_classifier
from data_generator import DataGenerator
from loss import contrastive_loss
from visualization import visualize_tsne

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

input_shape = (28, 28, 1)
pretrain_epochs = 20
classifier_epochs = 10
batch_size = 128

train_generator = DataGenerator(x_train, y_train, batch_size)
test_generator = DataGenerator(x_test, y_test, batch_size)

model = build_model(input_shape)
learning_rate = 0.0001
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss=lambda y_true, y_pred: contrastive_loss(y_true, y_pred, margin=0.5))

model.fit(train_generator, epochs=pretrain_epochs, validation_data=test_generator)

features_train = model.predict(x_train)
features_test = model.predict(x_test)

visualize_tsne(features_train, y_train)
visualize_tsne(features_test, y_test)

num_classes = 10
classifier = build_classifier(model, num_classes)
classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
classifier.fit(train_generator, epochs=classifier_epochs, validation_data=test_generator)

loss, accuracy = classifier.evaluate(test_generator)
print("Test accuracy: {:.2f}%".format(accuracy * 100))

# Save the trained classifier model
model_path = "mnist_classifier.h5"
classifier.save(model_path)
print(f"Model saved to {model_path}")
