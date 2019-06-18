from keras_vggface.vggface import VGGFace
from keras.layers import Flatten, Dense, Input,Dropout, Conv2D, MaxPool2D
from keras.models import Model
from keras import models
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import matplotlib.pyplot as plt

nb_class = 3
hidden_dim = 512
width_img = 256
height_img = 256

#1. Feature Extraction
vgg_model = VGGFace(weights='vggface', model='vgg16', include_top=False, input_shape=(width_img,height_img,3))

# list all the layer names which are in the model.
layer_names = [layer.name for layer in vgg_model.layers]
print("layers_name: ",layer_names)
for layer in vgg_model.layers:
    print(layer, layer.trainable)
    
# Create the model
model = models.Sequential()    
# Add the vgg convolutional base model
model.add(vgg_model)

#2. Finetuning
# Add new layers
model.add(Flatten())
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))
model.summary()

from keras import metrics
model.compile(optimizer = optimizers.SGD(lr=0.001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

# prepare the image augmentation 
train_datagen = ImageDataGenerator(
        rotation_range=30,
        # we will rescale all our pixel values between 0 and 1
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

test_datagen = ImageDataGenerator(
        rotation_range=30,
        # we will rescale all our pixel values between 0 and 1
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

training_set = train_datagen.flow_from_directory(
        'dataset/train_set',
        target_size=(width_img, height_img),
        batch_size=2,
        class_mode='categorical')
       
test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(width_img, height_img),
        batch_size=2,
        class_mode='categorical')

history = model.fit_generator(
        training_set,
        steps_per_epoch=11,
        validation_data=test_set,
        validation_steps=10,
        epochs=90, # number of loop
        )

model.save('model.h5')
print("Model saved !!")
classes = training_set.class_indices
# Phuc: 0,  Phuong : 1, Toan: 2
print(classes)

#%%plot accurancy and loss
#Let us see the loss and accuracy curves.
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))
 
plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig('chart/accuracy.png')
plt.figure()
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig('chart/loss.png')
plt.show()
