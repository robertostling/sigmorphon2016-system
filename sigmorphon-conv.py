"""Experiment with only 1D convolutions"""

import os
import pickle
import time
import copy
import sys

import numpy as np

from keras.models import Model, model_from_yaml
from keras.layers import (Embedding, Input, Activation, GRU, TimeDistributed,
                          Dense, merge, RepeatVector, LSTM, Convolution1D,
                          BatchNormalization, Dropout)
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.constraints import maxnorm
from keras.regularizers import l2


class FlushCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        print('Epoch %s: loss %s' % (epoch, logs.get('loss')), flush=True)


from data import MorphonData

SPECIAL_TASK2 = True
TESTING_MODE = 'TESTING' in sys.argv[1:]

class MorphonModel:
    def __init__(self, data, prefix, task):
        self.task = task
        self.data = data            # MorphonData instance
        self.prefix = prefix        # path of model files
        self.max_length = data.max_length + 4
        special_symbols = ['<PAD>', '<S>']
        self.alphabet = special_symbols + data.alphabet
        self.min_normal_symbol = len(special_symbols)
        self.alphabet_idx = {c:i for i,c in enumerate(self.alphabet)}


    def initialize(self):
        self.model.compile(optimizer='adam', loss='categorical_crossentropy')


    def build(self, embedding_dims=32, n_layers=1, rnn_layers=1,
              kernel_size=3, rnn_dims=128,
              use_bn=True, language=None, dropout=0, l2_penalty=None):
        conv_regularizer = None if l2_penalty is None else l2(l2_penalty)

        def smart_merge(vectors, **kwargs):
            return vectors[0] if len(vectors)==1 else merge(vectors, **kwargs)

        input_str = Input(shape=(self.max_length,), dtype='int32')
        input_feats = Input(shape=(self.data.feature_vector_length,),
                            dtype='float32')

        inputs = [input_str, input_feats]

        #if self.task == 2 and SPECIAL_TASK2:
        #    input_source_feats = Input(
        #            shape=(self.data.feature_vector_length,),
        #            dtype='int8')
        #    inputs.append(input_source_feats)
        #    time_invariant_vectors.append(input_source_feats)

        embeddings = Embedding(len(self.alphabet), embedding_dims,
                               input_length=self.max_length,
                               W_constraint=maxnorm(2))

        input_embedded = embeddings(input_str)

        conv_dims = embedding_dims + self.data.feature_vector_length
        layer = merge(
                [input_embedded, RepeatVector(self.max_length)(input_feats)],
                mode='concat')

        last_layer = layer
        for _ in range(n_layers):
            # http://arxiv.org/abs/1603.05027
            if use_bn:
                layer = BatchNormalization()(layer)
            layer = Activation('relu')(layer)
            if dropout:
                layer = Dropout(dropout)(layer)
            layer = Convolution1D(
                conv_dims, kernel_size, border_mode='same', init='he_normal',
                W_regularizer=conv_regularizer,
                b_regularizer=conv_regularizer)(layer)

            if use_bn:
                layer = BatchNormalization()(layer)
            layer = Activation('relu')(layer)
            if dropout:
                layer = Dropout(dropout)(layer)
            layer = Convolution1D(
                conv_dims, kernel_size, border_mode='same', init='he_normal',
                W_regularizer=conv_regularizer,
                b_regularizer=conv_regularizer)(layer)

            layer = merge([last_layer, layer], mode='sum')
            last_layer = layer

        if use_bn:
            layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        if dropout and rnn_layers == 0:
            layer = Dropout(dropout)(layer)

        for _ in range(rnn_layers):
            layer = GRU(rnn_dims, return_sequences=True, dropout_W=dropout,
                        consume_less='gpu')(layer)

        layer = TimeDistributed(Dense(len(self.alphabet)))(layer)
        outputs = Activation('softmax')(layer)

        self.model = Model(input=inputs, output=outputs)
        self.initialize()


    def is_saved(self):
        return     os.path.exists(self.prefix + '.yaml') \
               and os.path.exists(self.prefix + '.hdf5') \
               and os.path.exists(self.prefix + '.pkl')


    def load(self):
        with open(self.prefix + '.yaml', 'r', encoding='utf-8') as f:
            self.model = model_from_yaml(f.read())
        self.load_weights()
        self.initialize()


    def load_weights(self):
        self.model.load_weights(self.prefix + '.hdf5')


    def save(self, save_weights=True, save_structure=False, save_data=False):
        if save_structure:
            with open(self.prefix + '.yaml', 'w', encoding='utf-8') as f:
                f.write(self.model.to_yaml())
        if save_data:
            with open(self.prefix + '.pkl', 'wb') as f:
                pickle.dump(self.data, f, -1)
        if save_weights:
           self.model.save_weights(self.prefix + '.hdf5', overwrite=False)


    def encode_string(self, s, length, pad='<PAD>', backwards=False,
                      dtype=np.int8):
        assert length >= len(s)
        assert not backwards
        if backwards:
            return np.array(  [self.alphabet_idx[pad]]*(length-len(s))
                            + [self.alphabet_idx[c] for c in s[::-1].lower()],
                            dtype=dtype)
        else:
            return np.array(  [self.alphabet_idx[c] for c in s.lower()]
                            + [self.alphabet_idx[pad]]*(length-len(s)),
                            dtype=dtype)


    def decode_string(self, v):
        return ''.join(self.alphabet[x]
                       for x in v if x >= self.min_normal_symbol)


    def word_forms(self, task, part, target_feats=None, pos=None):
        wfs = [wf for wf in self.data.data[task-1][part]
               if     (   target_feats is None
                       or wf.target_feats == target_feats)
                  and (pos is None or wf.pos == pos)]
        return wfs[:500] if TESTING_MODE else wfs


    def make_source(self, word_forms, length=None, dtype=np.int8):
        if length is None: length = self.max_length
        vectors = [
                np.array([self.encode_string(wf.source, length, dtype=dtype)
                          for wf in word_forms],
                         dtype=dtype),
                np.array([self.data.encode_features(wf.target_feats_dict)
                          for wf in word_forms])]
        if self.task == 2 and SPECIAL_TASK2:
            vectors.append(
                np.array([self.data.encode_features(wf.source_feats_dict)
                          for wf in word_forms]))
        return vectors


    def make_target(self, word_forms, length=None, dtype=np.int8):
        if length is None: length = self.max_length
        return np.array([self.encode_string(wf.target, length, dtype=dtype)
                         for wf in word_forms],
                         dtype=dtype)


    def onehot(self, x, length=None):
        if length is None: length = len(self.alphabet)
        y = np.zeros(x.shape+(length,), dtype=np.int8)
        for i,row in enumerate(x):
            for j,col in enumerate(row):
                y[i,j,col] = 1
        return y


    def dataset(self, task, part, pos=None):
        word_forms = self.word_forms(task, part, pos=pos)
        x = self.make_source(word_forms)
        if part in ('train', 'dev', 'test'):
            y_sym = self.make_target(word_forms)
            y = self.onehot(y_sym)
            #y_shifted = np.hstack([
            #    np.full_like(y_sym[:,0:1], self.alphabet_idx['<S>']),
            #    y_sym[:,:-1]])
            y_strings = [wf.target for wf in word_forms]
            return (x, y, y_strings)
        else:
            return (x,)


    def predict(self, word_forms, beam_size=4):
        x = self.make_source(word_forms)
        y = self.model.predict(x).argmax(axis=-1)
        return [(wf.recapitalize(self.decode_string(row)), 0.0)
                for wf, row in zip(word_forms, y)]


    def write_test(self, f, part='test-covered'):
        word_forms = self.word_forms(self.task, part)
        strings, log_ps = list(zip(*self.predict(word_forms)))
        n_correct = 0
        n_total = 0
        for wf, s in zip(word_forms, strings):
            if not wf.target is None:
                n_total += 1
                n_correct += int(wf.target == s)
            wf_copy = copy.copy(wf)
            wf_copy.target = s
            print(str(wf_copy), file=f)
        if n_total:
            print('Accuracy: %d/%d = %.2f%%' % (
                n_correct, n_total, 100.0*n_correct/n_total))
            return n_correct/n_total

        #for wf, s in zip(word_forms, strings):
        #    wf_copy = copy.copy(wf)
        #    wf_copy.target = s
        #    print(str(wf_copy), file=f)


    def evaluate(self, part, verbose=1):
        word_forms = self.word_forms(self.task, part)
        strings, _ = list(zip(*self.predict(word_forms)))
        incorrect = [(pred, wf.target, wf.source, wf.target_feats)
                     for wf, pred in zip(word_forms, strings)
                     if wf.target != pred]
        accuracy = (len(word_forms) - len(incorrect)) / len(word_forms)
        if verbose >= 1:
            print('Accuracy: %.3f%%' % (100.0*accuracy), flush=True)
        if verbose >= 2:
            print('%d/%d incorrect classifications:' % (
                len(incorrect), len(word_forms)))
            for t in incorrect:
                print('  %-20s %-20s (%s, %s)' % t)
            print('-'*72, flush=True)

        return accuracy, strings


    def train(self, task, train_set=None, dev_set=None, pos=None):
        if train_set is None:
            train_set = self.dataset(task, 'train', pos=pos)
        if dev_set is None:
            dev_set = self.dataset(task, 'dev', pos=pos)
        self.model.fit(
                train_set[0], train_set[1],
                batch_size=128,
                nb_epoch=1 if TESTING_MODE else 10000,
                verbose=1 if TESTING_MODE else 2,
                validation_data=(dev_set[0], dev_set[1]),
                callbacks=[EarlyStopping(patience=4),
                           FlushCallback(),
                           ModelCheckpoint(self.prefix + '.hdf5',
                                           save_best_only=True,
                                           verbose=1)])



def train_model(prefix, data_path, task=1, **kwargs):
    if os.path.exists(prefix + '.pkl'):
        if    (not os.path.exists(prefix + '-solution')) \
           or (not os.path.exists(prefix + '-dev')):
            with open(prefix + '.pkl', 'rb') as f:
                data = pickle.load(f)
        else:
            print('Skipping %s' % prefix, flush=True)
            return None
    else:
        data = MorphonData(data_path)

    morphon = MorphonModel(data, prefix, task)
    if morphon.is_saved():
        print('Loading parameters from %s' % prefix, flush=True)
        morphon.load()
        fresh = False
    else:
        #kwargs.setdefault('conv_dims', kwargs['embedding_dims'])
        morphon.build(**kwargs)
        morphon.save(save_weights=False)
        morphon.train(task)
        # Load the weights of the best model, which was auto-saved during
        # training
        morphon.load_weights()
        fresh = True

    if not os.path.exists(prefix + '-solution'):
        print('Annotating test set...', flush=True)
        with open(prefix + '-solution', 'w', encoding='utf-8') as f:
            morphon.write_test(f, 'test-covered')

    acc = None
    if not os.path.exists(prefix + '-dev'):
        print('Annotating development set...', flush=True)
        with open(prefix + '-dev', 'w', encoding='utf-8') as f:
            acc = morphon.write_test(f, 'dev')

    #if fresh:
    #    print('Development set evaluation (%s)\n' % prefix)
    #    return morphon.evaluate('dev', verbose=1 if TESTING_MODE else 2)[0]
    return acc


def run_experiments(nr, languages):
    import glob

    data_path = '../sigmorphon2016/data'
    #all_languages = set(
    #                os.path.basename(filename).split('-')[0]
    #                for filename in glob.glob('%s/*-task1-train' % data_path))

    ablation_languages = languages # [] # ['finnish']
    #languages = ablation_languages + sorted(
    #        all_languages - set(ablation_languages))

    base_config = {
              'use_bn': True,
              'dropout': 0.5,
              'n_layers': 12,
              'rnn_layers': 0,
              'rnn_dims': 128,
              'kernel_size': 7,
              #'conv_dims': 128,
              'embedding_dims': 128,
              'l2_penalty': None }

    def generate_configurations(language, task, ablate=True):
        c = dict(base_config)
        c['task'] = task
        c['language'] = language

        yield dict(c)

        if not ablate: return

        ablations = [
            #('use_bn', [False]),
            #('dropout', [0.0]),
            #('n_conv_layers', [0]),
            #('use_encoder', [False]),
            #('share_embeddings', [False]),
            #('hidden_dims', [128, 64]),
            #('embedding_dims', [32, 16]),
            #('gate_name', ['gru']),
            #('n_layers', [8, 4]),
            #('rnn_layers', [0]),
            #('kernel_size', [5, 3])
            ]

        for k, vs in ablations:
            v0 = c[k]
            for v in vs:
                c[k] = v
                yield dict(c)
                # ---------
                c['rnn_layers'] = 0
                yield dict(c)
                c['rnn_layers'] = 1
                # ---------
            c[k] = v0

    configurations = [
            c for task in (3,)
              for language in languages
              for c in generate_configurations(
                  language, task, ablate=(language in ablation_languages))]

    model_base = './models/run%d' % nr

    with open(model_base + '.log', 'a', encoding='utf-8') as logf:
        for c in configurations:
            prefix = model_base + '-' + '-'.join(
                    '%s=%s' % kv for kv in sorted(c.items()))
            language = c['language']
            t0 = time.time()
            acc = train_model(prefix, '%s/%s' % (data_path, language), **c)
            t = time.time() - t0
            if acc is None: acc = 0.0
            print('%.3f\t%s\t%.2f\t%s' % (100.0*acc, language, t, prefix),
                  file=logf, flush=True)


if __name__ == '__main__':
    import sys
    run_experiments(int(sys.argv[1]),
                    [x for x in sys.argv[2:] if x != 'TESTING'])

