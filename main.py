import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    device = "/gpu:0"
else:
    device = "/cpu:0"

import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_MKL_REUSE_PRIMITIVE_MEMORY'] = '0'

import numpy as np
import pandas as pd
from tensorflow.keras import layers
from gnn import *
import nfp
import json 
import sys

from argparse import ArgumentParser
from tensorflow.keras.callbacks import ModelCheckpoint

from sklearn.model_selection import KFold
#import shap

def main(args):
    if args.fold_number != -1: #k-fold CV
        data = pd.read_csv('data_for_kfold_221003.csv')
        train = data[data.fold_num != args.fold_number]
        valid = data[data.fold_num == args.fold_number]
        test =  data[data.fold_num == args.fold_number] #dummy
    else:
        data = pd.read_csv('data_220913.csv')
        # just a random 8:1:1 split
        train = data.sample(frac=.8, random_state=args.random_seed)
        valid = data[~data.index.isin(train.index)].sample(frac=.5, random_state=args.random_seed)
        test = data[~data.index.isin(train.index) & ~data.index.isin(valid.index)]
    train['Train/Valid/Test'] = 'Train'
    valid['Train/Valid/Test'] = 'Valid'
    test['Train/Valid/Test'] = 'Test'

    preprocessor = CustomPreprocessor(
        explicit_hs=False,
        atom_features=atom_features,
        bond_features=bond_features)


    #!!!! modify output_signature, add mole fractions to the create_tf_dataset part !!!!!

    input_dict_signature = {'mol'+str(i):preprocessor.output_signature for i in range(1,14)}
    input_dict_signature['X'] = tf.TensorSpec(shape = [13], dtype=tf.float32)

    output_signature = (input_dict_signature,
                        tf.TensorSpec(shape=(), dtype=tf.float32),
                        tf.TensorSpec(shape=(), dtype=tf.float32))
 
    print(f"Atom classes before: {preprocessor.atom_classes} (includes 'none' and 'missing' classes)")
    print(f"Bond classes before: {preprocessor.bond_classes} (includes 'none' and 'missing' classes)")

    train_smiles = list(set(np.concatenate([list(train['can_smi_'+str(i)]) for i in range(1,14)])))

    for smiles in train_smiles:
        preprocessor.construct_feature_matrices(smiles, train=True)
    print(f'Atom classes after: {preprocessor.atom_classes}')
    print(f'Bond classes after: {preprocessor.bond_classes}')

    train_data = tf.data.Dataset.from_generator(
        lambda: create_tf_dataset(train, preprocessor, args.sample_weight, True), output_signature=output_signature)\
        .cache().shuffle(buffer_size=1000)\
        .padded_batch(batch_size=args.batchsize)\
        .prefetch(tf.data.experimental.AUTOTUNE)

    valid_data = tf.data.Dataset.from_generator(
        lambda: create_tf_dataset(valid, preprocessor, args.sample_weight, False), output_signature=output_signature)\
        .cache()\
        .padded_batch(batch_size=args.batchsize)\
        .prefetch(tf.data.experimental.AUTOTUNE)

    test_data = tf.data.Dataset.from_generator(
        lambda: create_tf_dataset(test, preprocessor, args.sample_weight, False), output_signature=output_signature)\
        .cache()\
        .padded_batch(batch_size=args.batchsize)\
        .prefetch(tf.data.experimental.AUTOTUNE)

    model = CN_blend_model(args.num_hidden, args.layers, preprocessor, args.no_glob_feat)
    model.compile(loss='mae', optimizer=tf.keras.optimizers.Adam(args.lr))

    #model_path = "model_files/"+args.modelname+"/best_model.h5"
    model_path = "model_files/"+args.modelname+"/best_model.ckpt"

    checkpoint = ModelCheckpoint(model_path, monitor="val_loss",\
                                 verbose=2, save_best_only = True, 
                                 save_weights_only = True,
                                 mode='auto', period=1 )

    hist = model.fit(train_data,
                     validation_data=valid_data,
                     epochs=args.epoch,
                     verbose=2, callbacks = [checkpoint])

    model.load_weights(model_path)

    train_data_final = tf.data.Dataset.from_generator(
        lambda: create_tf_dataset(train, preprocessor, args.sample_weight, False), output_signature=output_signature)\
        .cache()\
        .padded_batch(batch_size=args.batchsize)\
        .prefetch(tf.data.experimental.AUTOTUNE)

    train_results = model.predict(train_data_final).squeeze()
    valid_results = model.predict(valid_data).squeeze()

    mae_train = np.abs(train_results - train['CN']).mean()
    mae_valid = np.abs(valid_results - valid['CN']).mean()

    if args.fold_number == -1:
        test_results = model.predict(test_data).squeeze()
        mae_test = np.abs(test_results - test['CN']).mean()
        print(len(train),len(valid),len(test))
        print(mae_train,mae_valid,mae_test)
        train['predicted'] = train_results
        valid['predicted'] = valid_results
        test['predicted'] = test_results
        pd.concat([train, valid, test], ignore_index=True).to_csv('model_files/' + args.modelname +'/results.csv',index=False)
    else:
        print("Fold number", args.fold_number)
        print(len(train),len(valid))
        print(mae_train,mae_valid)

        train['predicted'] = train_results
        valid['predicted'] = valid_results

        pd.concat([train, valid], ignore_index=True).to_csv('model_files/' + args.modelname +'/kfold_'+str(args.fold_number)+'.csv',index=False)

    preprocessor.to_json("model_files/"+ args.modelname  +"/preprocessor.json")

if __name__ == '__main__':
    with tf.device(device):
        parser = ArgumentParser()
        parser.add_argument('-lr', type=float, default=1.0e-4, help='Learning rate (default=1.0e-4)')
        parser.add_argument('-batchsize', type=int, default=16, help='batch_size (default=16)')
        parser.add_argument('-epoch', type=int, default=1000, help='epoch (default=1000)')
        parser.add_argument('-layers', type=int, default=5, help='number of gnn layers (default=5)')
        parser.add_argument('-num_hidden', type=int, default=64, help='number of nodes in hidden layers (default=64)')
        parser.add_argument('-random_seed', type=int, default=1, help='random seed number used when splitting the dataset (default=1)')
        parser.add_argument('-sample_weight', type=float, default=0.6, help='whether to use sample weights (default=0.6) If 1.0 -> no sample weights, if < 1.0 -> sample weights to Tier 2,3 methods')
        parser.add_argument('-no_glob_feat', action='store_true', default=False, help='If specified, no global features/updates are used (default=Use global features)')

        parser.add_argument('-modelname', type=str, default='', help='model name (default=blank)')
        parser.add_argument('-fold_number', type=int, default=-1, help='fold number for Kfold')
        args = parser.parse_args()
    main(args)
