from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Model

from tensorflow.keras.layers import BatchNormalization

def build_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    model = Model(inputs, x)
    return model

def build_classifier(base_model, num_classes):
    x = base_model.output
    x = Dense(128, activation='relu')(x)
    x = Dense(num_classes, activation='softmax')(x)
    classifier = Model(base_model.input, x)
    return classifier



