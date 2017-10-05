import argparse
import logging
import sys
import os
import glob
import matplotlib.pyplot as plt

from keras import  __version__
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD

def get_nb_files(directory):

    if not os.path.exists(directory):
        return 0

    cnt = 0

    for r, dirs, files, in os.walk(directory):
        for dr in dirs:
            cnt += len(glob.glob(os.path.join(r, dr + "/*")))

    return cnt

def add_new_last_layer(base_model, nb_classes, args):

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(args.fc_size, activation="relu")(x)
    predictions = Dense(nb_classes, activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def setup_to_transfer_learning(model, base_model):

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer="rmsprop", loss="categorical_crossentropy",
                  metrics=["accuracy"])

def setup_to_finetune(model, args):

    for layer in model.layers[:args.nb_iv3_layers_to_freeze]:
        layer.trainable = False
    for layer in model.layers[args.nb_iv3_layers_to_freeze:]:
        layer.trainable = True

    model.compile(optimizer=SGD(lr=0.001, momentum=0.9), loss="categorical_crossentropy", metrics=["accuracy"])

def plot_training(history):

    acc = history.history["acc"]
    val_acc = history.history["val_acc"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r.')
    plt.plot(epochs, val_acc, 'r')
    plt.title("Training and validation accuracy")

    plt.figure()
    plt.plot(epochs, loss, 'r.')
    plt.plot(epochs, val_loss, 'r-')
    plt.title("Training and validation loss")
    plt.show()

def train(args):

    nb_train_samples = get_nb_files(args.train_dir)
    logging.info("Number of training samples: {}".format(nb_train_samples))

    nb_classes = len(glob.glob(args.train_dir + "/*"))
    logging.info("Number of classes: {}".format(nb_classes))

    nb_val_samples = get_nb_files(args.val_dir)
    logging.info("Number of validation samples: {}".format(nb_val_samples))

    nb_epoch = int(args.nb_epoch)
    logging.info("Number of epochs: {}".format(nb_epoch))

    batch_size = int(args.batch_size)
    logging.info("Batch size: {}".format(batch_size))

    # data prep
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    train_generator = train_datagen.flow_from_directory(
        args.train_dir,
        target_size=(args.im_width, args.im_height),
        batch_size=args.batch_size,
    )

    validation_generator = test_datagen.flow_from_directory(
        args.train_dir,
        target_size=(args.im_width, args.im_height),
        batch_size=args.batch_size,
    )

    # setup model
    base_model = InceptionV3(weights='imagenet', include_top=False)
    model = add_new_last_layer(base_model, nb_classes, args)

    # transfer learning
    setup_to_transfer_learning(model, base_model)

    history_tl = model.fit_generator(
        train_generator,
        steps_per_epoch= int(nb_train_samples // batch_size),
        epochs=nb_epoch,
        validation_data=validation_generator,
        validation_steps= nb_val_samples // batch_size,
        class_weight="auto"
    )

    # fine-tuning
    setup_to_finetune(model, args)

    history_tl = model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=nb_epoch,
        validation_data=validation_generator,
        validation_steps=nb_val_samples // batch_size,
        class_weight="auto"
    )

    model.save(args.output_model_file)

    if args.plot:
        plot_training(history_tl)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, default="/home/deeplearning/data/train_dir")
    parser.add_argument("--val_dir", type=str, default="/home/deeplearning/data/val_dir")
    parser.add_argument("--nb_epoch", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--fc_size", type=int, default=1024)
    parser.add_argument("--nb_iv3_layers_to_freeze", type=int, default=172)
    parser.add_argument("--output_model_file", type=str, default="inceptionv3-ft.model")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--im_width", type=int, default=299)
    parser.add_argument("--im_height", type=int, default=299)

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)

    if args.train_dir is None or args.val_dir is None:
        parser.print_help()
        sys.exit(1)

    if (not os.path.exists(args.train_dir)) or (not os.path.exists(args.val_dir)):
        print("directories do not exist")
        sys.exit(1)

    train(args)
