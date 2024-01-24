from keras.preprocessing.image import ImageDataGenerator

aug = ImageDataGenerator(
    rotation_range = 30,
    rescale=1/255.0
)

target_size = (24,24)
batch_size = 32
train_data_gen = aug.flow_from_directory(
     'data/train',
     target_size = target_size,
     batch_size = batch_size,
     color_mode='grayscale'
)

test_data_gen = aug.flow_from_directory(
     'data/test',
     target_size = target_size,
     batch_size = batch_size,
     color_mode='grayscale'
)

import keras

model = keras.models.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu',input_shape=(24,24,1)),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Conv2D(128, (3,3), activation='relu'),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Flatten(),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(2, activation='softmax')
])

epochs=15; lr=1e-3
from keras.optimizers.legacy import Adam

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer = Adam(learning_rate=lr, decay=lr/epochs),
              metrics=['accuracy'])

print("Training the model")

model.fit(
        train_data_gen,
        validation_data = test_data_gen,
        epochs = epochs
    )

model.save('models/my_model.h5', overwrite=True)

