import pandas as pd

import torch

import os
import glob

import pytorch_lightning as pl

from models.seq2seq import Seq2SeqModule
from models.vae import VqVaeModule

from datasets import dataset_type, DataSetType

from change_vocab import change_size_inout_layer_new_chord_vocab, get_new_chord_quality_vocab, \
    NEW_FIGARO_CHECKPOINT_PATH, NEW_VAE_CHECKPOINT_PATH, change_size_desc_layer_new_chord_vocab

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ROOT_DIR = os.getenv('ROOT_DIR', './lmd_full')
OUTPUT_DIR = os.getenv('OUTPUT_DIR', './results')
LOGGING_DIR = os.getenv('LOGGING_DIR', './logs')
MAX_N_FILES = int(os.getenv('MAX_N_FILES', -1))

MODEL = os.getenv('MODEL', None)
MODEL_NAME = os.getenv('MODEL_NAME', None)
N_CODES = int(os.getenv('N_CODES', 2048))
N_GROUPS = int(os.getenv('N_GROUPS', 16))
D_MODEL = int(os.getenv('D_MODEL', 512))
D_LATENT = int(os.getenv('D_LATENT', 1024))

CHECKPOINT = os.getenv('CHECKPOINT', None)
VAE_CHECKPOINT = os.getenv('VAE_CHECKPOINT', None)

BATCH_SIZE = int(os.getenv('BATCH_SIZE', 128))
TARGET_BATCH_SIZE = int(os.getenv('TARGET_BATCH_SIZE', 512))

EPOCHS = int(os.getenv('EPOCHS', '16'))
WARMUP_STEPS = int(float(os.getenv('WARMUP_STEPS', 4000)))
MAX_STEPS = int(float(os.getenv('MAX_STEPS', 1e20)))
MAX_TRAINING_STEPS = int(float(os.getenv('MAX_TRAINING_STEPS', 100_000)))
LEARNING_RATE = float(os.getenv('LEARNING_RATE', 1e-4))
LR_SCHEDULE = os.getenv('LR_SCHEDULE', 'const')
CONTEXT_SIZE = int(os.getenv('CONTEXT_SIZE', 256))

ACCUMULATE_GRADS = max(1, TARGET_BATCH_SIZE // BATCH_SIZE)

N_WORKERS = min(os.cpu_count(), float(os.getenv('N_WORKERS', 'inf')))
if device.type == 'cuda':
    N_WORKERS = min(N_WORKERS, 8 * torch.cuda.device_count())
N_WORKERS = int(N_WORKERS)

MY_CHECKPOINT_DIR = '../checkpoints/my_checkpoint/'


def main():
    ### Define available models ###

    available_models = [
        'vq-vae',
        'figaro-learned',
        'figaro-expert',
        'figaro',
        'figaro-inst',
        'figaro-chord',
        'figaro-meta',
        'figaro-no-inst',
        'figaro-no-chord',
        'figaro-no-meta',
        'baseline',
    ]

    MODEL = 'figaro'  # fixme: correct model?

    assert MODEL is not None, 'the MODEL needs to be specified'
    assert MODEL in available_models, f'unknown MODEL: {MODEL}'

    ### Create data loaders ###
    # input_files = glob.glob(os.path.join(ROOT_DIR, '**/*.mid'), recursive=True)
    if dataset_type == DataSetType.MIDI:
        # input_files = glob.glob('../my_midi_dataset/*.mid', recursive=True)
        input_files = glob.glob('../my_midi_dataset/alex/*.midi', recursive=True)
        if MAX_N_FILES > 0:
            input_files = input_files[:MAX_N_FILES]
    elif dataset_type == DataSetType.JSON:
        df = pd.read_csv('../my_csv_dataset/less_read_100_new.csv')
        input_files = df.values.tolist()
    else:
        print(f"WARNING: Incorrect Dataset_type{dataset_type}")
        exit()

    if len(input_files) == 0:
        print(
            f"WARNING: No MIDI files were found at '{ROOT_DIR}'. Did you download the dataset to the right location?")
        exit()

    MAX_CONTEXT = min(1024, CONTEXT_SIZE)

    VAE_CHECKPOINT = "../checkpoints/vq-vae.ckpt"  # fixme correct path
    # CHECKPOINT = '../checkpoints/figaro.ckpt'
    CHECKPOINT = NEW_FIGARO_CHECKPOINT_PATH # fixme

    if MODEL in ['figaro-learned', 'figaro'] and VAE_CHECKPOINT:
        vae_module = VqVaeModule.load_from_checkpoint(checkpoint_path=VAE_CHECKPOINT)
        vae_module.cpu()
        vae_module.freeze()
        vae_module.eval()

    else:
        vae_module = None

    ### Create and train model ###

    # load model from checkpoint if available


    if CHECKPOINT:
        model_class = {
            'vq-vae': VqVaeModule,
            'figaro-learned': Seq2SeqModule,
            'figaro-expert': Seq2SeqModule,
            'figaro': Seq2SeqModule,
            'figaro-inst': Seq2SeqModule,
            'figaro-chord': Seq2SeqModule,
            'figaro-meta': Seq2SeqModule,
            'figaro-no-inst': Seq2SeqModule,
            'figaro-no-chord': Seq2SeqModule,
            'figaro-no-meta': Seq2SeqModule,
            'baseline': Seq2SeqModule,
        }[MODEL]
        # new_qualities = ['sus', 'power', '6']
        # change_size_new_chord_vocab(module=model_class, new_qualities=new_qualities)
        model = model_class.load_from_checkpoint(checkpoint_path=CHECKPOINT)

    else:
        seq2seq_kwargs = {
            'encoder_layers': 4,
            'decoder_layers': 6,
            'num_attention_heads': 8,
            'intermediate_size': 2048,
            'd_model': D_MODEL,
            'context_size': MAX_CONTEXT,
            'lr': LEARNING_RATE,
            'warmup_steps': WARMUP_STEPS,
            'max_steps': MAX_STEPS,
        }
        dec_kwargs = {**seq2seq_kwargs}
        dec_kwargs['encoder_layers'] = 0

        # use lambda functions for lazy initialization
        model = {
            'vq-vae': lambda: VqVaeModule(
                encoder_layers=4,
                decoder_layers=6,
                encoder_ffn_dim=2048,
                decoder_ffn_dim=2048,
                n_codes=N_CODES,
                n_groups=N_GROUPS,
                context_size=MAX_CONTEXT,
                lr=LEARNING_RATE,
                lr_schedule=LR_SCHEDULE,
                warmup_steps=WARMUP_STEPS,
                max_steps=MAX_STEPS,
                d_model=D_MODEL,
                d_latent=D_LATENT,
            ),
            'figaro-learned': lambda: Seq2SeqModule(
                description_flavor='latent',
                n_codes=vae_module.n_codes,
                n_groups=vae_module.n_groups,
                d_latent=vae_module.d_latent,
                **seq2seq_kwargs
            ),
            'figaro': lambda: Seq2SeqModule(
                description_flavor='both',
                n_codes=vae_module.n_codes,
                n_groups=vae_module.n_groups,
                d_latent=vae_module.d_latent,
                **seq2seq_kwargs
            ),
            'figaro-expert': lambda: Seq2SeqModule(
                description_flavor='description',
                **seq2seq_kwargs
            ),
            'figaro-no-meta': lambda: Seq2SeqModule(
                description_flavor='description',
                description_options={'instruments': True, 'chords': True, 'meta': False},
                **seq2seq_kwargs
            ),
            'figaro-no-inst': lambda: Seq2SeqModule(
                description_flavor='description',
                description_options={'instruments': False, 'chords': True, 'meta': True},
                **seq2seq_kwargs
            ),
            'figaro-no-chord': lambda: Seq2SeqModule(
                description_flavor='description',
                description_options={'instruments': True, 'chords': False, 'meta': True},
                **seq2seq_kwargs
            ),
            'baseline': lambda: Seq2SeqModule(
                description_flavor='none',
                **dec_kwargs
            ),
        }[MODEL]()

    datamodule = model.get_datamodule(
        input_files,
        vae_module=vae_module,
        batch_size=BATCH_SIZE,
        num_workers=N_WORKERS,
        pin_memory=True
    )

    checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(
        monitor='valid_loss',
        # dirpath=os.path.join(OUTPUT_DIR, MODEL),
        dirpath=os.path.join('../checkpoints/my_checkpoint', MODEL),
        filename='{step}-{valid_loss:.2f}',
        save_last=True,
        save_top_k=2,
        every_n_train_steps=1000,
        # every_n_train_steps=1,
    )

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(
        gpus=0 if device.type == 'cpu' else torch.cuda.device_count(),
        accelerator='dp',
        profiler='simple',
        callbacks=[checkpoint_callback, lr_monitor],
        max_epochs=EPOCHS,
        max_steps=MAX_TRAINING_STEPS,
        log_every_n_steps=max(100, min(25 * ACCUMULATE_GRADS, 200)),
        val_check_interval=max(500, min(300 * ACCUMULATE_GRADS, 1000)),
        limit_val_batches=64,
        auto_scale_batch_size=False,
        auto_lr_find=False,
        accumulate_grad_batches=ACCUMULATE_GRADS,
        stochastic_weight_avg=True,
        gradient_clip_val=1.0,
        terminate_on_nan=True,
        resume_from_checkpoint=CHECKPOINT,
    )

    # used to adjust the model size after chord vocabulary extension
    new_qualities = ['sus', 'pwr', '6']

    trainer.fit(model, datamodule)

    # change_size_desc_layer_new_chord_vocab(module=model, new_qualities=new_qualities)
    # change_size_inout_layer_new_chord_vocab(module=model, new_qualities=new_qualities)
    # change_size_inout_layer_new_chord_vocab(module=vae_module, new_qualities=new_qualities)

    trainer.save_checkpoint(NEW_FIGARO_CHECKPOINT_PATH)

    debug_stop = 1


if __name__ == '__main__':
    main()
