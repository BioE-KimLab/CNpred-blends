import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    tf.config.experimental.set_memory_growth(gpus[0], True)

import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_MKL_REUSE_PRIMITIVE_MEMORY'] = '0'

import numpy as np
from collections import namedtuple

from tensorflow.keras import layers
import nfp

import rdkit.Chem
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem.rdMolDescriptors import CalcNumHBA, CalcNumHBD
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import pandas as pd
import json

class CustomPreprocessor(nfp.SmilesPreprocessor):
    def construct_feature_matrices(self, smiles, train=None):
        features = super(CustomPreprocessor, self).construct_feature_matrices(smiles, train)
        features['mol_features'] = global_features(smiles)
        return features
    
    output_signature = {**nfp.SmilesPreprocessor.output_signature,
                     **{'mol_features': tf.TensorSpec(shape=(2,), dtype=tf.float32) }}

def atom_features(atom):
    atom_type = namedtuple('Atom', ['totalHs', 'symbol', 'aromatic', 'ring_size'])
    return str((atom.GetTotalNumHs(),
                atom.GetSymbol(),
                atom.GetIsAromatic(),
                nfp.preprocessing.features.get_ring_size(atom, max_size=6)
               ))

def bond_features(bond, flipped=False):
    bond_type = namedtuple('Bond', ['bond_type', 'ring_size', 'symbol_1', 'symbol_2'])

    if not flipped:
        atom1 = bond.GetBeginAtom()
        atom2 = bond.GetEndAtom()

    else:
        atom1 = bond.GetEndAtom()
        atom2 = bond.GetBeginAtom()

    return str((bond.GetBondType(),
                nfp.preprocessing.features.get_ring_size(bond, max_size=6),
                atom1.GetSymbol(),
                atom2.GetSymbol()
               ))

def global_features(smiles):
    mol = rdkit.Chem.MolFromSmiles(smiles)

    return tf.constant([CalcNumHBA(mol),
                     CalcNumHBD(mol)])

def create_tf_dataset(df, preprocessor, sample_weight = 1.0, train=True):
    input_dicts = {} 
    for _, row in df.iterrows():
        for component_i in range(1,14):
            inputs = preprocessor.construct_feature_matrices(row['can_smi_'+str(component_i)], train=train)
            #inputs = preprocessor.construct_feature_matrices(row['Canonical_SMILES'], train=train)

            one_data_sample_w = 1.0
            '''
            if not train:
                one_data_sample_w = 1.0
            elif sample_weight < 1.0:
                if row['Device_tier'] == 1:
                    one_data_sample_w = 1.0
                else:
                    one_data_sample_w = sample_weight
            else:
                one_data_sample_w = 1.0
            '''

            input_dicts['mol'+str(component_i)] = {'atom': inputs['atom'],
                                                 'bond': inputs['bond'],
                                                 'connectivity': inputs['connectivity'],
                                                 'mol_features': global_features(row['can_smi_'+str(component_i)])}
        input_dicts['X'] = [ float(row['x_mol_'+str(component_i)]) for component_i in range(1,14) ]

        yield (input_dicts, row['CN'], one_data_sample_w)


def message_block(original_atom_state, original_bond_state,
                 original_global_state, connectivity, features_dim, i):
    
    atom_state = original_atom_state
    bond_state = original_bond_state
    global_state = original_global_state
    
    global_state_update = layers.GlobalAveragePooling1D()(atom_state)
    global_state_update = layers.Dense(features_dim, activation='relu')(global_state_update)
    global_state_update = layers.Dense(features_dim)(global_state_update)
    global_state = layers.Add()([original_global_state, global_state_update])
    
    new_bond_state = nfp.EdgeUpdate()([atom_state, bond_state, connectivity, global_state])
    bond_state = layers.Add()([original_bond_state, new_bond_state])
    
    new_atom_state = nfp.NodeUpdate()([atom_state, bond_state, connectivity, global_state])
    atom_state = layers.Add()([original_atom_state, new_atom_state])
    
    return atom_state, bond_state, global_state

def message_block_no_glob(original_atom_state, original_bond_state, connectivity, features_dim, i):
    atom_state = original_atom_state
    bond_state = original_bond_state

    new_bond_state = nfp.EdgeUpdate()([atom_state, bond_state, connectivity])
    bond_state = layers.Add()([original_bond_state, new_bond_state])

    new_atom_state = nfp.NodeUpdate()([atom_state, bond_state, connectivity])
    atom_state = layers.Add()([original_atom_state, new_atom_state])

    return atom_state, bond_state


def build_model(features_dim, num_messages, preprocessor, no_glob_feat):
    atom_Input = layers.Input(shape=[None], dtype=tf.int32, name='atom')
    bond_Input = layers.Input(shape=[None], dtype=tf.int32, name='bond')
    connectivity_Input = layers.Input(shape=[None, 2], dtype=tf.int32, name='connectivity')
    global_Input = layers.Input(shape=[2], dtype=tf.float32, name='mol_features')


    atom_state = layers.Embedding(preprocessor.atom_classes, features_dim,
                                  name='atom_embedding', mask_zero=True,
                                  embeddings_regularizer='l2')(atom_Input)
    bond_state = layers.Embedding(preprocessor.bond_classes, features_dim,
                                  name='bond_embedding', mask_zero=True,
                                  embeddings_regularizer='l2')(bond_Input)

    if no_glob_feat:
        atom_mean = layers.Embedding(preprocessor.atom_classes, 1,
                                     name='atom_mean', mask_zero=True,
                                     embeddings_regularizer='l2')(atom_Input)

        for i in range(num_messages):
            atom_state, bond_state = message_block_no_glob(atom_state, bond_state,
                                                        connectivity_Input, features_dim, i)

        atom_state = layers.Add()([atom_state, atom_mean])
        atom_state = layers.Dense(1)(atom_state)

        prediction = layers.GlobalAveragePooling1D()(atom_state)

        input_tensors = [atom_Input, bond_Input, connectivity_Input]
    else:
        global_state = layers.Dense(features_dim, activation='relu')(global_Input) 

        for i in range(num_messages):
            atom_state, bond_state, global_state = message_block(atom_state, bond_state,
                                                                 global_state, connectivity_Input, features_dim, i)

        #prediction = layers.Dense(1)(global_state)
        prediction = layers.Dense(4)(global_state)

        input_tensors = [atom_Input, bond_Input, connectivity_Input, global_Input]

    model = tf.keras.Model(input_tensors, [prediction])
    return model


class CN_blend_model(tf.keras.Model):
    def __init__(self,features_dim, num_messages, preprocessor, no_glob_feat):
        super().__init__()
        self.gnn_model = build_model(features_dim, num_messages, preprocessor, no_glob_feat)
        #self.gnn_model.load_weights('model_files/2_sw6/best_model.h5')

    def call(self, inputs):
        UVs = [self.gnn_model(inputs['mol'+str(i)]) for i in range(1,14)] # (num_data * num_components * 4)
        U, V = tf.split(UVs, num_or_size_splits = 2, axis = -1) # (num_data * num_components * 2)
        U = tf.transpose(U, perm=[1,0,2])
        V = tf.transpose(V, perm=[1,0,2])

        W = tf.linalg.matmul(U, V, transpose_b = True)

        X = inputs['X']
        XW = tf.matmul(a = tf.expand_dims(X, axis=1), b = W)

        CNs = tf.reduce_sum( tf.math.multiply(X, tf.squeeze(XW, axis=1)), axis = -1 )
        #print(CNs.shape)
        return CNs 
