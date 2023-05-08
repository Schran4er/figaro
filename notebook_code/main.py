from descriptions import simple_description

import os

import logging
import math
import re
import time
from collections import Counter

import IPython
import numpy as np
import pretty_midi
import pytorch_lightning as pl
import soundfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext
import transformers

import tqdm.notebook as tqdm # here
# from tqdm import tqdm

from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    BertConfig,
    EncoderDecoderConfig,
    EncoderDecoderModel
)

import fluidsynth

pretty_midi.instrument._HAS_FLUIDSYNTH = True
pretty_midi.instrument.fluidsynth = fluidsynth

# FIXME: Defining helper functions

# parameters for input representation
DEFAULT_POS_PER_QUARTER = 12
DEFAULT_VELOCITY_BINS = np.linspace(0, 128, 32 + 1, dtype=np)
DEFAULT_DURATION_BINS = np.sort(np.concatenate([
    np.arange(1, 13),  # smallest possible units up to 1 quarter
    np.arange(12, 24, 3)[1:],  # 16th notes up to 1 bar
    np.arange(13, 24, 4)[1:],  # triplets up to 1 bar
    np.arange(24, 48, 6),  # 8th notes up to 2 bars
    np.arange(48, 4 * 48, 12),  # quarter notes up to 8 bars
    np.arange(4 * 48, 16 * 48 + 1, 24)  # half notes up to 16 bars
]))
DEFAULT_TEMPO_BINS = np.linspace(0, 240, 32 + 1, dtype=np)
DEFAULT_NOTE_DENSITY_BINS = np.linspace(0, 12, 32 + 1)
DEFAULT_MEAN_VELOCITY_BINS = np.linspace(0, 128, 32 + 1)
DEFAULT_MEAN_PITCH_BINS = np.linspace(0, 128, 32 + 1)
DEFAULT_MEAN_DURATION_BINS = np.logspace(0, 7, 32 + 1, base=2)  # log space between 1 and 128 positions (~2.5 bars)

# parameters for output
DEFAULT_RESOLUTION = 480

# maximum length of a single bar is 3*4 = 12 beats
MAX_BAR_LENGTH = 3
# maximum number of bars in a piece is 512 (this covers almost all sequences)
MAX_N_BARS = 512

PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
BOS_TOKEN = '<bos>'
EOS_TOKEN = '<eos>'
MASK_TOKEN = '<mask>'

TIME_SIGNATURE_KEY = 'Time Signature'
BAR_KEY = 'Bar'
POSITION_KEY = 'Position'
INSTRUMENT_KEY = 'Instrument'
PITCH_KEY = 'Pitch'
VELOCITY_KEY = 'Velocity'
DURATION_KEY = 'Duration'
TEMPO_KEY = 'Tempo'
CHORD_KEY = 'Chord'

NOTE_DENSITY_KEY = 'Note Density'
MEAN_PITCH_KEY = 'Mean Pitch'
MEAN_VELOCITY_KEY = 'Mean Velocity'
MEAN_DURATION_KEY = 'Mean Duration'


class Tokens:
    def get_instrument_tokens(key=INSTRUMENT_KEY):
        tokens = [f'{key}_{pretty_midi.program_to_instrument_name(i)}' for i in range(128)]
        tokens.append(f'{key}_drum')
        return tokens

    def get_chord_tokens(key=CHORD_KEY, qualities=['maj', 'min', 'dim', 'aug', 'dom7', 'maj7', 'min7', 'None']):
        pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

        chords = [f'{root}:{quality}' for root in pitch_classes for quality in qualities]
        chords.append('N:N')

        tokens = [f'{key}_{chord}' for chord in chords]
        return tokens

    def get_time_signature_tokens(key=TIME_SIGNATURE_KEY):
        denominators = [2, 4, 8, 16]
        time_sigs = [f'{p}/{q}' for q in denominators for p in range(1, MAX_BAR_LENGTH * q + 1)]
        tokens = [f'{key}_{time_sig}' for time_sig in time_sigs]
        return tokens

    def get_midi_tokens(
            instrument_key=INSTRUMENT_KEY,
            time_signature_key=TIME_SIGNATURE_KEY,
            pitch_key=PITCH_KEY,
            velocity_key=VELOCITY_KEY,
            duration_key=DURATION_KEY,
            tempo_key=TEMPO_KEY,
            bar_key=BAR_KEY,
            position_key=POSITION_KEY
    ):
        instrument_tokens = Tokens.get_instrument_tokens(instrument_key)

        pitch_tokens = [f'{pitch_key}_{i}' for i in range(128)] + [f'{pitch_key}_drum_{i}' for i in range(128)]
        velocity_tokens = [f'{velocity_key}_{i}' for i in range(len(DEFAULT_VELOCITY_BINS))]
        duration_tokens = [f'{duration_key}_{i}' for i in range(len(DEFAULT_DURATION_BINS))]
        tempo_tokens = [f'{tempo_key}_{i}' for i in range(len(DEFAULT_TEMPO_BINS))]
        bar_tokens = [f'{bar_key}_{i}' for i in range(MAX_N_BARS)]
        position_tokens = [f'{position_key}_{i}' for i in range(MAX_BAR_LENGTH * 4 * DEFAULT_POS_PER_QUARTER)]

        time_sig_tokens = Tokens.get_time_signature_tokens(time_signature_key)

        return (
                time_sig_tokens +
                tempo_tokens +
                instrument_tokens +
                pitch_tokens +
                velocity_tokens +
                duration_tokens +
                bar_tokens +
                position_tokens
        )


class Vocab:
    def __init__(self, counter, specials=[PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, MASK_TOKEN], unk_token=UNK_TOKEN):
        self.vocab = torchtext.vocab.vocab(counter)

        self.specials = specials
        for i, token in enumerate(self.specials):
            self.vocab.insert_token(token, i)

        if unk_token in specials:
            self.vocab.set_default_index(self.vocab.get_stoi()[unk_token])

    def to_i(self, token):
        return self.vocab.get_stoi()[token]

    def to_s(self, idx):
        if idx >= len(self.vocab):
            return UNK_TOKEN
        else:
            return self.vocab.get_itos()[idx]

    def __len__(self):
        return len(self.vocab)

    def encode(self, seq):
        return self.vocab(seq)

    def decode(self, seq):
        if isinstance(seq, Tensor):
            seq = seq.numpy()
        return self.vocab.lookup_tokens(seq)


class RemiVocab(Vocab):
    def __init__(self):
        midi_tokens = Tokens.get_midi_tokens()
        chord_tokens = Tokens.get_chord_tokens()

        self.tokens = midi_tokens + chord_tokens

        counter = Counter(self.tokens)
        super().__init__(counter)


class DescriptionVocab(Vocab):
    def __init__(self):
        time_sig_tokens = Tokens.get_time_signature_tokens()
        instrument_tokens = Tokens.get_instrument_tokens()
        chord_tokens = Tokens.get_chord_tokens()

        bar_tokens = [f'Bar_{i}' for i in range(MAX_N_BARS)]
        density_tokens = [f'{NOTE_DENSITY_KEY}_{i}' for i in range(len(DEFAULT_NOTE_DENSITY_BINS))]
        velocity_tokens = [f'{MEAN_VELOCITY_KEY}_{i}' for i in range(len(DEFAULT_MEAN_VELOCITY_BINS))]
        pitch_tokens = [f'{MEAN_PITCH_KEY}_{i}' for i in range(len(DEFAULT_MEAN_PITCH_BINS))]
        duration_tokens = [f'{MEAN_DURATION_KEY}_{i}' for i in range(len(DEFAULT_MEAN_DURATION_BINS))]

        self.tokens = (
                time_sig_tokens +
                instrument_tokens +
                chord_tokens +
                density_tokens +
                velocity_tokens +
                pitch_tokens +
                duration_tokens +
                bar_tokens
        )

        counter = Counter(self.tokens)
        super().__init__(counter)


class GroupEmbedding(nn.Module):
    def __init__(self, n_tokens, n_groups, out_dim, inner_dim=128):
        super().__init__()
        self.n_tokens = n_tokens
        self.n_groups = n_groups
        self.inner_dim = inner_dim
        self.out_dim = out_dim

        self.embedding = nn.Embedding(n_tokens, inner_dim)
        self.proj = nn.Linear(n_groups * inner_dim, out_dim, bias=False)

    def forward(self, x):
        shape = x.shape
        emb = self.embedding(x)
        return self.proj(emb.view(*shape[:-1], self.n_groups * self.inner_dim))


class Seq2SeqModule(pl.LightningModule):
    def __init__(self,
                 d_model=512,
                 d_latent=512,
                 n_codes=512,
                 n_groups=8,
                 context_size=512,
                 lr=1e-4,
                 lr_schedule='sqrt_decay',
                 warmup_steps=None,
                 max_steps=None,
                 encoder_layers=6,
                 decoder_layers=12,
                 intermediate_size=2048,
                 num_attention_heads=8,
                 description_flavor='description',
                 description_options=None,
                 use_pretrained_latent_embeddings=True):
        super(Seq2SeqModule, self).__init__()

        self.description_flavor = description_flavor
        assert self.description_flavor in ['latent', 'description', 'none',
                                           'both'], f"Unknown description flavor '{self.description_flavor}', expected one of ['latent', 'description', 'none', 'both]"
        self.description_options = description_options

        self.context_size = context_size
        self.d_model = d_model
        self.d_latent = d_latent

        self.lr = lr
        self.lr_schedule = lr_schedule
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps

        self.vocab = RemiVocab()

        encoder_config = BertConfig(
            vocab_size=1,
            pad_token_id=0,
            hidden_size=self.d_model,
            num_hidden_layers=encoder_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=1024,
            position_embedding_type='relative_key_query'
        )
        decoder_config = BertConfig(
            vocab_size=1,
            pad_token_id=0,
            hidden_size=self.d_model,
            num_hidden_layers=decoder_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=1024,
            position_embedding_type='relative_key_query'
        )
        config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, decoder_config)
        self.transformer = EncoderDecoderModel(config)
        self.transformer.config.decoder.is_decoder = True
        self.transformer.config.decoder.add_cross_attention = True

        self.max_bars = self.context_size
        self.max_positions = 512
        self.bar_embedding = nn.Embedding(self.max_bars + 1, self.d_model)
        self.pos_embedding = nn.Embedding(self.max_positions + 1, self.d_model)

        if self.description_flavor in ['latent', 'both']:
            if use_pretrained_latent_embeddings:
                self.latent_in = nn.Linear(self.d_latent, self.d_model, bias=False)
            else:
                self.latent_in = GroupEmbedding(n_codes, n_groups, self.d_model, inner_dim=self.d_latent // n_groups)
        if self.description_flavor in ['description', 'both']:
            desc_vocab = DescriptionVocab()
            self.desc_in = nn.Embedding(len(desc_vocab), self.d_model)

        if self.description_flavor == 'both':
            self.desc_proj = nn.Linear(2 * self.d_model, self.d_model, bias=False)

        self.in_layer = nn.Embedding(len(self.vocab), self.d_model)
        self.out_layer = nn.Linear(self.d_model, len(self.vocab), bias=False)

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.vocab.to_i(PAD_TOKEN))

        self.save_hyperparameters()

    def encode(self, z, desc_bar_ids=None):
        if self.description_flavor == 'both':
            desc = z['description']
            latent = z['latents']
            desc_emb = self.desc_in(desc)
            latent_emb = self.latent_in(latent)

            padded = pad_sequence([desc_emb.transpose(0, 1), latent_emb.transpose(0, 1)], batch_first=True)
            desc_emb, latent_emb = padded.transpose(1, 2)

            if desc_bar_ids is not None:
                # Use the fact that description is always longer than latents
                desc_emb = desc_emb + self.bar_embedding(desc_bar_ids)

            z_emb = self.desc_proj(torch.cat([desc_emb, latent_emb], dim=-1))

        elif self.description_flavor == 'description':
            z_emb = self.desc_in(z)
            if desc_bar_ids is not None:
                z_emb += self.bar_embedding(desc_bar_ids)

        elif self.description_flavor == 'latent':
            z_emb = self.latent_in(z)

        else:
            return None

        out = self.transformer.encoder(inputs_embeds=z_emb, output_hidden_states=True)
        encoder_hidden = out.hidden_states[-1]
        return encoder_hidden

    def decode(self, x, labels=None, bar_ids=None, position_ids=None, encoder_hidden_states=None, return_hidden=False):
        seq_len = x.size(1)

        # Shape of x_emb: (batch_size, seq_len, d_model)
        x_emb = self.in_layer(x)
        if bar_ids is not None:
            x_emb += self.bar_embedding(bar_ids)
        if position_ids is not None:
            x_emb += self.pos_embedding(position_ids)

        if encoder_hidden_states is not None:
            # Make x_emb and encoder_hidden_states match in sequence length. Necessary for relative positional embeddings
            padded = pad_sequence([x_emb.transpose(0, 1), encoder_hidden_states.transpose(0, 1)], batch_first=True)
            x_emb, encoder_hidden_states = padded.transpose(1, 2)

            out = self.transformer.decoder(
                inputs_embeds=x_emb,
                encoder_hidden_states=encoder_hidden_states,
                output_hidden_states=True
            )
            hidden = out.hidden_states[-1][:, :seq_len]
        else:
            out = self.transformer.decoder(inputs_embeds=x_emb, output_hidden_states=True)
            hidden = out.hidden_states[-1][:, :seq_len]

        # Shape of logits: (batch_size, seq_len, tuple_size, vocab_size)

        if return_hidden:
            return hidden
        else:
            return self.out_layer(hidden)

    def forward(self, x, z=None, labels=None, position_ids=None, bar_ids=None, description_bar_ids=None,
                return_hidden=False):
        encoder_hidden = self.encode(z, desc_bar_ids=description_bar_ids)

        out = self.decode(x,
                          labels=labels,
                          bar_ids=bar_ids,
                          position_ids=position_ids,
                          encoder_hidden_states=encoder_hidden,
                          return_hidden=return_hidden
                          )

        return out

    def get_loss(self, batch, return_logits=False):
        # Shape of x: (batch_size, seq_len, tuple_size)
        x = batch['input_ids']
        bar_ids = batch['bar_ids']
        position_ids = batch['position_ids']
        # Shape of labels: (batch_size, tgt_len, tuple_size)
        labels = batch['labels']

        # Shape of z: (batch_size, context_size, n_groups, d_latent)
        if self.description_flavor == 'latent':
            z = batch['latents']
            desc_bar_ids = None
        elif self.description_flavor == 'description':
            z = batch['description']
            desc_bar_ids = batch['desc_bar_ids']
        elif self.description_flavor == 'both':
            z = {'latents': batch['latents'], 'description': batch['description']}
            desc_bar_ids = batch['desc_bar_ids']
        else:
            z, desc_bar_ids = None, None

        logits = self(x, z=z, labels=labels, bar_ids=bar_ids, position_ids=position_ids,
                      description_bar_ids=desc_bar_ids)
        # Shape of logits: (batch_size, tgt_len, tuple_size, vocab_size)
        pred = logits.view(-1, logits.shape[-1])
        labels = labels.reshape(-1)

        loss = self.loss_fn(pred, labels)

        if return_logits:
            return loss, logits
        else:
            return loss

    def training_step(self, batch, batch_idx):
        loss = self.get_loss(batch)
        self.log('train_loss', loss.detach(), on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits = self.get_loss(batch, return_logits=True)
        self.log('valid_loss', loss.detach(), on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

        y = batch['labels']
        pad_token_id = self.vocab.to_i(PAD_TOKEN)

        logits = logits.view(logits.size(0), -1, logits.size(-1))
        y = y.view(y.size(0), -1)

        log_pr = logits.log_softmax(dim=-1)
        log_pr[y == pad_token_id] = 0  # log(pr) = log(1) for padding
        log_pr = torch.gather(log_pr, -1, y.unsqueeze(-1)).squeeze(-1)

        t = (y != pad_token_id).sum(dim=-1)
        ppl = (-log_pr.sum(dim=1) / t).exp().mean()
        self.log('valid_ppl', ppl.detach(), on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        return self.get_loss(batch)

    def configure_optimizers(self):
        # set LR to 1, scale with LambdaLR scheduler
        optimizer = transformers.AdamW(self.parameters(), lr=1, weight_decay=0.01)

        if self.lr_schedule == 'sqrt_decay':
            # constant warmup, then 1/sqrt(n) decay starting from the initial LR
            lr_func = lambda step: min(self.lr, self.lr / math.sqrt(max(step, 1) / self.warmup_steps))
        elif self.lr_schedule == 'linear':
            # linear warmup, linear decay
            lr_func = lambda step: min(self.lr, self.lr * step / self.warmup_steps,
                                       self.lr * (1 - (step - self.warmup_steps) / self.max_steps))
        elif self.lr_schedule == 'cosine':
            # linear warmup, cosine decay to 10% of initial LR
            lr_func = lambda step: self.lr * min(step / self.warmup_steps, 0.55 + 0.45 * math.cos(
                math.pi * (min(step, self.max_steps) - self.warmup_steps) / (self.max_steps - self.warmup_steps)))
        else:
            # Use no lr scheduling
            lr_func = lambda step: self.lr

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)
        return [optimizer], [{
            'scheduler': scheduler,
            'interval': 'step',
        }]

    @torch.no_grad()
    def sample(self, batch,
               max_length=256,
               max_bars=-1,
               temp=0.8,
               pad_token=PAD_TOKEN,
               eos_token=EOS_TOKEN,
               verbose=0,
               ):

        # Setup and parsing arguments

        pad_token_id = self.vocab.to_i(pad_token)
        eos_token_id = self.vocab.to_i(eos_token)

        batch_size, curr_len = batch['input_ids'].shape

        i = curr_len - 1

        x = batch['input_ids']
        bar_ids = batch['bar_ids']
        position_ids = batch['position_ids']
        assert x.shape[:2] == bar_ids.shape and x.shape[
                                                :2] == position_ids.shape, f"Input, bar and position ids weren't of compatible shapes: {x.shape}, {bar_ids.shape}, {position_ids.shape}"

        if self.description_flavor == 'both':
            z = {'latents': batch['latents'], 'description': batch['description']}
            desc_bar_ids = batch['desc_bar_ids']
        elif self.description_flavor == 'latent':
            z, desc_bar_ids = batch['latents'], None
        elif self.description_flavor == 'description':
            z, desc_bar_ids = batch['description'], batch['desc_bar_ids']
        else:
            z, desc_bar_ids = None, None

        is_done = torch.zeros(batch_size, dtype=torch.bool, device=x.device)

        # Precompute encoder hidden states for cross-attention
        if self.description_flavor == 'latent':
            encoder_hidden_states = self.encode(z, desc_bar_ids)
        else:
            encoder_hidden_states = None

        curr_bars = torch.zeros(batch_size, dtype=torch.int, device=x.device).fill_(-1)
        # Sample using decoder until max_length is reached or all sequences are done
        for iter in tqdm.trange(curr_len - 1, max_length, desc="Generating tokens", smoothing=0.01):
            i = x.size(1) - 1
            # print(f"\r{i+1}/{max_length}", end='')
            x_ = x[:, -self.context_size:].to(self.device)
            bar_ids_ = bar_ids[:, -self.context_size:].to(self.device)
            position_ids_ = position_ids[:, -self.context_size:].to(self.device)

            # Description scrolling
            if self.description_flavor in ['description', 'both']:
                if self.description_flavor == 'description':
                    desc = z
                else:
                    desc = z['description']

                next_bars = bar_ids[:, -self.context_size:][:, 0]
                bars_changed = not (next_bars == curr_bars).all()
                curr_bars = next_bars

                if bars_changed:
                    z_ = torch.zeros(batch_size, self.context_size, dtype=torch.int)
                    desc_bar_ids_ = torch.zeros(batch_size, self.context_size, dtype=torch.int)

                    for j in range(batch_size):
                        curr_bar = curr_bars[j]
                        indices = torch.nonzero(desc_bar_ids[j] == curr_bar)
                        if indices.size(0) > 0:
                            idx = indices[0, 0]
                        else:
                            idx = desc.size(1) - 1

                        offset = min(self.context_size, desc.size(1) - idx)

                        z_[j, :offset] = desc[j, idx:idx + offset]
                        desc_bar_ids_[j, :offset] = desc_bar_ids[j, idx:idx + offset]

                    z_, desc_bar_ids_ = z_.to(self.device), desc_bar_ids_.to(self.device)

                    if self.description_flavor == 'both':
                        z_ = {'description': z_, 'latents': z['latents']}

                    encoder_hidden_states = self.encode(z_, desc_bar_ids_)

            logits = self.decode(x_, bar_ids=bar_ids_, position_ids=position_ids_,
                                 encoder_hidden_states=encoder_hidden_states)

            next_token_ids = torch.zeros((batch_size,), dtype=torch.int, device=x.device)
            for batch_idx in range(batch_size):
                idx = min(self.context_size - 1, i)
                scores = logits[batch_idx, idx]

                while True:
                    pr = (scores / temp).softmax(dim=-1).view(-1)

                    token_id = torch.multinomial(pr, 1).view(-1).to(x.device)

                    next_x = torch.cat([x[batch_idx], token_id])
                    if next_x.size(0) >= 10 and (next_x[-5:] == next_x[-10:-5]).all():
                        if verbose:
                            logging.warning("WARNING: sampled token is invalid, masking this token and sampling again")
                        scores[token_id[0]] = -float("inf")
                        continue

                    next_token_ids[batch_idx] = token_id[0]
                    break

            next_tokens = self.vocab.decode(next_token_ids)

            if verbose:
                print(f"{i + 1}/{max_length}", next_tokens)

            next_bars = torch.tensor([1 if f'{BAR_KEY}_' in token else 0 for token in next_tokens], dtype=torch.int)
            next_bar_ids = bar_ids[:, i].clone() + next_bars

            next_positions = [f"{POSITION_KEY}_0" if f'{BAR_KEY}_' in token else token for token in next_tokens]
            next_positions = [int(token.split('_')[-1]) if f'{POSITION_KEY}_' in token else None for token in
                              next_positions]
            next_positions = [pos if next_pos is None else next_pos for pos, next_pos in
                              zip(position_ids[:, i], next_positions)]
            next_position_ids = torch.tensor(next_positions, dtype=torch.int)

            is_done.masked_fill_((next_token_ids == eos_token_id).all(dim=-1), True)
            next_token_ids[is_done] = pad_token_id
            if max_bars > 0:
                is_done.masked_fill_(next_bar_ids >= max_bars + 1, True)

            x = torch.cat([x, next_token_ids.clone().unsqueeze(1)], dim=1)
            bar_ids = torch.cat([bar_ids, next_bar_ids.unsqueeze(1)], dim=1)
            position_ids = torch.cat([position_ids, next_position_ids.unsqueeze(1)], dim=1)

            if torch.all(is_done):
                break

            # check if the model is repeating itself
            if x.size(1) >= 10 and (x[:, -5:] == x[:, -10:-5]).all():
                if verbose:
                    logging.warning(
                        "WARNING: model is repeating itself and producing invalid sequences, removing the invalid tokens")
                x = x[:, :-5]
                bar_ids = bar_ids[:, :-5]
                position_ids = position_ids[:, :-5]

        return {
            'sequences': x,
            'bar_ids': bar_ids,
            'position_ids': position_ids
        }


def remi2midi(events, bpm=120, time_signature=(4, 4), polyphony_limit=16):
    vocab = RemiVocab()

    def _get_time(bar, position, bpm=120, positions_per_bar=48):
        abs_position = bar * positions_per_bar + position
        beat = abs_position / DEFAULT_POS_PER_QUARTER
        return beat / bpm * 60

    def _get_time(reference, bar, pos):
        time_sig = reference['time_sig']
        num, denom = time_sig.numerator, time_sig.denominator
        # Quarters per bar, assuming 4 quarters per whole note
        qpb = 4 * num / denom
        ref_pos = reference['pos']
        d_bars = bar - ref_pos[0]
        d_pos = (pos - ref_pos[1]) + d_bars * qpb * DEFAULT_POS_PER_QUARTER
        d_quarters = d_pos / DEFAULT_POS_PER_QUARTER
        # Convert quarters to seconds
        dt = d_quarters / reference['tempo'] * 60
        return reference['time'] + dt

    # time_sigs = [event.split('_')[-1].split('/') for event in events if f"{TIME_SIGNATURE_KEY}_" in event]
    # time_sigs = [(int(num), int(denom)) for num, denom in time_sigs]

    tempo_changes = [event for event in events if f"{TEMPO_KEY}_" in event]
    if len(tempo_changes) > 0:
        bpm = DEFAULT_TEMPO_BINS[int(tempo_changes[0].split('_')[-1])]

    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    num, denom = time_signature
    pm.time_signature_changes.append(pretty_midi.TimeSignature(num, denom, 0))
    current_time_sig = pm.time_signature_changes[0]

    instruments = {}

    # Use implicit timeline: keep track of last tempo/time signature change event
    # and calculate time difference relative to that
    last_tl_event = {
        'time': 0,
        'pos': (0, 0),
        'time_sig': current_time_sig,
        'tempo': bpm
    }

    bar = -1
    n_notes = 0
    polyphony_control = {}
    for i, event in enumerate(events):
        if event == EOS_TOKEN:
            break

        if not bar in polyphony_control:
            polyphony_control[bar] = {}

        if f"{BAR_KEY}_" in events[i]:
            # Next bar is starting
            bar += 1
            polyphony_control[bar] = {}

            if i + 1 < len(events) and f"{TIME_SIGNATURE_KEY}_" in events[i + 1]:
                num, denom = events[i + 1].split('_')[-1].split('/')
                num, denom = int(num), int(denom)
                current_time_sig = last_tl_event['time_sig']
                if num != current_time_sig.numerator or denom != current_time_sig.denominator:
                    time = _get_time(last_tl_event, bar, 0)
                    time_sig = pretty_midi.TimeSignature(num, denom, time)
                    pm.time_signature_changes.append(time_sig)
                    last_tl_event['time'] = time
                    last_tl_event['pos'] = (bar, 0)
                    last_tl_event['time_sig'] = time_sig

        elif i + 1 < len(events) and \
                f"{POSITION_KEY}_" in events[i] and \
                f"{TEMPO_KEY}_" in events[i + 1]:
            position = int(events[i].split('_')[-1])
            tempo_idx = int(events[i + 1].split('_')[-1])
            tempo = DEFAULT_TEMPO_BINS[tempo_idx]

            if tempo != last_tl_event['tempo']:
                time = _get_time(last_tl_event, bar, position)
                last_tl_event['time'] = time
                last_tl_event['pos'] = (bar, position)
                # don't change the tempo throughout the piece
                # last_tl_event['tempo'] = tempo

        elif i + 4 < len(events) and \
                f"{POSITION_KEY}_" in events[i] and \
                f"{INSTRUMENT_KEY}_" in events[i + 1] and \
                f"{PITCH_KEY}_" in events[i + 2] and \
                f"{VELOCITY_KEY}_" in events[i + 3] and \
                f"{DURATION_KEY}_" in events[i + 4]:
            # get position
            position = int(events[i].split('_')[-1])
            if not position in polyphony_control[bar]:
                polyphony_control[bar][position] = {}

            # get instrument
            instrument_name = events[i + 1].split('_')[-1]
            if instrument_name not in polyphony_control[bar][position]:
                polyphony_control[bar][position][instrument_name] = 0
            elif polyphony_control[bar][position][instrument_name] >= polyphony_limit:
                # If number of notes exceeds polyphony limit, omit this note
                continue

            if instrument_name not in instruments:
                if instrument_name == 'drum':
                    instrument = pretty_midi.Instrument(0, is_drum=True)
                else:
                    program = pretty_midi.instrument_name_to_program(instrument_name)
                    instrument = pretty_midi.Instrument(program)
                instruments[instrument_name] = instrument
            else:
                instrument = instruments[instrument_name]

            # get pitch
            pitch = int(events[i + 2].split('_')[-1])
            # get velocity
            velocity_index = int(events[i + 3].split('_')[-1])
            velocity = int(min(127, DEFAULT_VELOCITY_BINS[velocity_index]))
                    # cast to int for pretty_midi "track.append(mido.Message(
                    # 'note_on', time=self.time_to_tick(note.start),
                    # channel=channel, note=note.pitch, velocity=note.velocity))"
            # get duration
            duration_index = int(events[i + 4].split('_')[-1])
            duration = DEFAULT_DURATION_BINS[duration_index]
            # create not and add to instrument
            start = _get_time(last_tl_event, bar, position)
            end = _get_time(last_tl_event, bar, position + duration)
            note = pretty_midi.Note(velocity=velocity,
                                    pitch=pitch,
                                    start=start,
                                    end=end)
            instrument.notes.append(note)
            n_notes += 1
            polyphony_control[bar][position][instrument_name] += 1

    for instrument in instruments.values():
        pm.instruments.append(instrument)
    return pm


desc_vocab = DescriptionVocab()
remi_vocab = RemiVocab()


def preprocess_description(desc, desc_vocab=desc_vocab):
    desc = "\n".join(re.findall(r"<[^>]+>", desc.strip()))
    desc = re.sub(r"[<>]", "", desc)
    desc = desc.replace("_Drums", "_drum")
    desc = desc.split("\n")
    check_description(desc, desc_vocab=desc_vocab)
    return desc


def check_description(desc, desc_vocab=desc_vocab):
    desc_ids = desc_vocab.encode(desc)
    tokens = desc_vocab.decode(desc_ids)
    if len(desc) != len(tokens):
        logging.error("Number of tokens was different after decoding, not sure what happened.")
    for desc_token, decoded_token in zip(desc, tokens):
        if desc_token != decoded_token:
            logging.error(f"Unable to encode token '{desc_token}' (was encoded to '{decoded_token}')")

    # TODO: check if the description is valid
    # check if it has the right order (meta tokens -> instruments -> chords)
    # check if it has the right meta tokens in the right order


def estimate_number_of_tokens(desc, desc_vocab=desc_vocab):
    desc_events = preprocess_description(desc, desc_vocab=desc_vocab)
    time_signatures = [tuple(int(x) for x in event.split("_")[-1].split("/")) for event in desc_events if
                       f"{TIME_SIGNATURE_KEY}_" in event]
    note_densities = [int(event.split("_")[-1]) for event in desc_events if f"{NOTE_DENSITY_KEY}_" in event]
    num_quarters = [num / denom * 4 for num, denom in time_signatures]
    densities = [DEFAULT_NOTE_DENSITY_BINS[d] for d in note_densities]

    expected_notes = DEFAULT_POS_PER_QUARTER * sum(q * d for q, d in zip(num_quarters, densities))
    n_bars = len(time_signatures)

    # it takes 5 tokens for every note, each bar usually has 6 token at the beginning
    return 5 * expected_notes + 6 * n_bars


def make_example_from_description(description: str, desc_vocab=desc_vocab, remi_vocab=remi_vocab):
    desc_events = preprocess_description(description, desc_vocab=desc_vocab)
    desc_bars = [i for i, event in enumerate(desc_events) if f"{BAR_KEY}_" in event]
    assert len(desc_bars) < 512, "The maximum number of allowed bars is 511."

    desc_bar_ids = torch.zeros(len(desc_events), dtype=torch.int)
    desc_bar_ids[desc_bars] = 1
    desc_bar_ids = torch.cumsum(desc_bar_ids, dim=0)

    zero = torch.tensor([0], dtype=torch.int)

    desc_ids = torch.tensor(desc_vocab.encode([BOS_TOKEN] + desc_events + [EOS_TOKEN]), dtype=torch.int)
    desc_bar_ids = torch.cat([zero, desc_bar_ids, zero])

    input_ids = torch.tensor(remi_vocab.encode([BOS_TOKEN]), dtype=torch.int)
    position_ids = torch.tensor([0], dtype=torch.int)
    bar_ids = torch.tensor([0], dtype=torch.int)

    return {
        "description": desc_ids,
        "desc_bar_ids": desc_bar_ids,
        "input_ids": input_ids,
        "position_ids": position_ids,
        "bar_ids": bar_ids,
    }


def generate_sample_from_description(description):
    # parse the given description to model input
    print("asd")
    n_bars = len(re.findall(r"<Bar_[0-9]+>", description))
    example = make_example_from_description(description)
    batch = {key: tensor.unsqueeze(0) for key, tensor in example.items()}

    # print description and expected number of notes
    approx_n_tokens = estimate_number_of_tokens(description)
    print(
        f"The generated sample based on this description will contain approximately {approx_n_tokens:.0f} tokens (~{int((approx_n_tokens - 6 * n_bars) / 5)} notes)")
    print("=== Description ===")
    print(description.strip())
    print("===")

    # generate a sample based on the description
    sample = model.sample(batch, max_bars=n_bars, max_length=int(approx_n_tokens * 1.2))

    # convert the generated tokens to MIDI and write it to disk
    remi_events = remi_vocab.decode(sample["sequences"][0])
    pm = remi2midi(remi_events)
    pm.write("sample_3.mid")


    # synthesize the generated MIDI and display it: # TODO: synthesize
    audio = pm.fluidsynth()
    soundfile.write("sample.wav", audio, 44100)
    return IPython.display.Audio("sample.wav")




if __name__ == '__main__':
    print("started")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # FIXME: loading the model

    model = Seq2SeqModule.load_from_checkpoint("checkpoints/figaro-expert.ckpt")
    model.freeze()
    model.eval()
    model.to(device)

    generate_sample_from_description(simple_description)
