import sys
import os
import argparse

import tensorflow as tf
import numpy as np

import pickle

sys.path.append(os.getcwd()+'\\machine_learning')
sys.path.append(os.getcwd()+'\\networks')

from networks import net3
from machine_learning import data_iterator
from machine_learning import utils

parser = argparse.ArgumentParser(description='TF2 train flags')
parser.add_argument('--csv_dir_path',       default=os.getcwd() + '\\DATA\\',        type=str, help='Directory path storing label csv files')
parser.add_argument('--model_save_path',    default=os.getcwd() + '\\saved_model\\', type=str, help='Dir path to save model weights')
parser.add_argument('--pretrain_max_epoch', default=70,                              type=int, help='Number of pretrain epochs')
parser.add_argument('--retrain_max_epoch',  default=70,                              type=int, help='Number of retrain epochs')

args = parser.parse_args()


public_train_info, public_valid_info=data_iterator.read_data(path=args.csv_dir_path,
                                                             whichset='public')
Public_train_iter=data_iterator.Data_Iterator(public_train_info)
Public_valid_iter=data_iterator.Data_Iterator(public_valid_info)
# labels, sexs, images = Public_iter.GetData()

model = net3.generate_model(drop_out_rate=0.4)
model.compile(loss="mse",
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
              metrics=["mae"])


@tf.function
def tf_training(train_labels, train_sexs, train_images):
    loss = model.train_on_batch(x=[train_images, train_sexs],
                                y=train_labels)
    return loss


@tf.function
def tf_validating(test_labels, test_sexs, test_images):
    loss = model.test_on_batch(x=[test_images, test_sexs],
                               y=test_labels)
    return loss

def validate(data_iter,
             current_epoch,
             weight_save_name):
    validation_loss = []
    print("Validating...")

    # iterate validation data
    while(True):

        test_labels, test_sexs, test_images = data_iter.GetData()
        test_images = test_images.reshape(-1, 79, 95, 79, 1)
        l = tf_validating(test_labels, test_sexs, test_images)
        utils.print_process(data_iter, current_epoch)
        validation_loss.append(l)

        if (data_iter.current_pointer == 0):
            break

    validation_loss = np.array(validation_loss)
    print("\n Validation MSE: {:10}; MAE: {:10}".format(np.round(validation_loss.mean(0)[0], 5),
                                                        np.round(validation_loss.mean(0)[1], 5)))

    # save model weights into pickle file
    weight_list = []
    for idx in range(len(model.weights)):
        weight_list.append(model.weights[idx].numpy())
    with open(weight_save_name + "_{}_{}.pickle".format(current_epoch,
                                                        np.round(validation_loss.mean(0)[1], 0)), "wb") as fp:
        pickle.dump(weight_list, fp)

def train(data_iter,
         current_epoch):
    training_loss = []
    print("Training...")

    # iterate training data
    while(True):
        train_labels, train_sexs, train_images = data_iter.GetData()
        train_images = train_images.reshape(-1, 79, 95, 79, 1)
        l = tf_training(train_labels, train_sexs, train_images)

        utils.print_process(data_iter, current_epoch)
        training_loss.append(l)

        if (data_iter.current_pointer == 0):
            break

    training_loss = np.array(training_loss)
    print("\n Training MSE: {:10}; MAE: {:10}".format(np.round(training_loss.mean(0)[0], 5),
                                                      np.round(training_loss.mean(0)[1], 5)))

# pretrain
for epoch in range(args.pretrain_max_epoch):
    # VALIDATION
    if (Public_train_iter.current_pointer == 0):
        validate(data_iter=Public_valid_iter,
                 current_epoch=epoch,
                 weight_save_name=args.model_save_path + 'model_pre')

    # TRAIN
    train(data_iter=Public_train_iter,
             current_epoch=epoch)



ih_train_info, ih_valid_info=data_iterator.read_data(path=args.csv_dir_path,
                                                     whichset='inhouse')

IH_train_iter=data_iterator.Data_Iterator(ih_train_info)
IH_valid_iter=data_iterator.Data_Iterator(ih_valid_info)


# retrain
for epoch in range(args.retrain_max_epoch):
    # VALIDATION
    if (IH_train_iter.current_pointer == 0):
        validate(data_iter=IH_valid_iter,
                 current_epoch=epoch,
                 weight_save_name=args.model_save_path + 'model_re')

    # TRAIN
    train(data_iter=IH_train_iter,
          current_epoch=epoch)