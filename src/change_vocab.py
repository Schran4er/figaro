import inspect

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchtext
from collections import Counter

from models.vae import VqVaeModule
from src.constants import MAX_N_BARS, NOTE_DENSITY_KEY, DEFAULT_NOTE_DENSITY_BINS, DEFAULT_MEAN_PITCH_BINS, \
    DEFAULT_MEAN_DURATION_BINS, DEFAULT_MEAN_VELOCITY_BINS, MEAN_VELOCITY_KEY, MEAN_PITCH_KEY, MEAN_DURATION_KEY, \
    PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, MASK_TOKEN, CHORD_KEY
from src.models.seq2seq import Seq2SeqModule
from vocab import Tokens, RemiVocab

NEW_FIGARO_CHECKPOINT_PATH = '../checkpoints/my_new_figaro_checkpoint.ckpt'
NEW_VAE_CHECKPOINT_PATH = '../checkpoints/my_new_vae_checkpoint.ckpt'


def get_len_with_chord_qualities(new_chord_qualities: list):
    # the following code fragments are pieced together from the constructor of class RemiVocab in vocab.py
    midi_tokens = Tokens.get_midi_tokens()
    new_chord_tokens = Tokens.get_chord_tokens(
        qualities=new_chord_qualities)
    tokens = midi_tokens + new_chord_tokens
    counter = Counter(tokens)
    new_vocab = torchtext.vocab.vocab(counter)

    specials = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, MASK_TOKEN]
    for i, token in enumerate(specials):
        new_vocab.insert_token(token, i)

    if UNK_TOKEN in specials:
        new_vocab.set_default_index(new_vocab.get_stoi()[UNK_TOKEN])

    return len(new_vocab)


def get_desc_len_with_chord_qualities(new_chord_qualities: list):
    time_sig_tokens = Tokens.get_time_signature_tokens()
    instrument_tokens = Tokens.get_instrument_tokens()
    chord_tokens = Tokens.get_chord_tokens()

    bar_tokens = [f'Bar_{i}' for i in range(MAX_N_BARS)]
    density_tokens = [f'{NOTE_DENSITY_KEY}_{i}' for i in range(len(DEFAULT_NOTE_DENSITY_BINS))]
    velocity_tokens = [f'{MEAN_VELOCITY_KEY}_{i}' for i in range(len(DEFAULT_MEAN_VELOCITY_BINS))]
    pitch_tokens = [f'{MEAN_PITCH_KEY}_{i}' for i in range(len(DEFAULT_MEAN_PITCH_BINS))]
    duration_tokens = [f'{MEAN_DURATION_KEY}_{i}' for i in range(len(DEFAULT_MEAN_DURATION_BINS))]

    specials = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, MASK_TOKEN]
    new_chord_tokens = get_new_chord_quality_vocab(new_qualities=new_chord_qualities) # because the last one is Chord_N:N which already exists

    tokens = (
            specials +
            time_sig_tokens +
            instrument_tokens +
            chord_tokens +
            density_tokens +
            velocity_tokens +
            pitch_tokens +
            duration_tokens +
            bar_tokens +
            new_chord_tokens
    )

    return len(tokens)

def change_size_desc_layer_new_chord_vocab(module, new_qualities):
    new_desc_size = get_desc_len_with_chord_qualities(new_chord_qualities=new_qualities)

    sig_module = inspect.signature(Seq2SeqModule)
    d_model = sig_module.parameters['d_model'].default

    desc_in = module.desc_in
    old_desc_in = desc_in.weight.data
    new_desc_in = nn.Embedding(new_desc_size, d_model).weight.data
    new_desc_in[:old_desc_in.size(0)] = old_desc_in
    with torch.no_grad():
        desc_in.weight.set_(new_desc_in)
    module.desc_in.num_embeddings = new_desc_size
    a = 1 # fixme



def change_size_inout_layer_new_chord_vocab(module, new_qualities):
    sig_chord_tokens = inspect.signature(Tokens.get_chord_tokens)
    old_chord_qualities = sig_chord_tokens.parameters['qualities'].default

    new_chord_qualities = old_chord_qualities + new_qualities
    new_vocab_size = get_len_with_chord_qualities(new_chord_qualities=new_chord_qualities)

    sig_module = inspect.signature(VqVaeModule)
    d_model = sig_module.parameters['d_model'].default

    # VAE_CHECKPOINT = "../checkpoints/vq-vae.ckpt"
    # vae_module = VqVaeModule.load_from_checkpoint(checkpoint_path=VAE_CHECKPOINT)

    in_layer = module.in_layer
    old_weights_in_layer = in_layer.weight.data
    new_weights_in_layer = nn.Embedding(new_vocab_size, d_model).weight.data
    new_weights_in_layer[:old_weights_in_layer.size(0)] = old_weights_in_layer
    with torch.no_grad():
        in_layer.weight.set_(new_weights_in_layer)
    module.in_layer.num_embeddings = new_vocab_size

    out_layer = module.out_layer
    old_weights_out_layer = out_layer.weight.data
    new_weights_out_layer = nn.Linear(d_model, new_vocab_size).weight.data # todo: this should be Linear? # ERROR
    new_weights_out_layer[:old_weights_out_layer.size(0)] = old_weights_out_layer
    with torch.no_grad():
        out_layer.weight.set_(new_weights_out_layer)
    # todo?
    # out_layer.out_features = new_vocab_size
    # out_layer.num_embeddings = new_vocab_size

    module.vocab = RemiVocab(update_vocab=True)


def get_new_chord_quality_vocab(new_qualities):
    pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    new_chord_qualities = [f'{root}:{quality}' for root in pitch_classes for quality in new_qualities]
    # new_chord_quality_vocab = [f'{CHORD_KEY}_{chord}' for chord in new_chord_qualities]
    new_chord_quality_vocab = [f'CHORD_{chord}' for chord in new_chord_qualities]
    return new_chord_quality_vocab
    # return new_chord_qualities


if __name__ == '__main__':
    a = 1

