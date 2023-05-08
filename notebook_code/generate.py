import logging
import sys, os
import re

import IPython
import soundfile
import torch
from dotenv import load_dotenv, set_key

from constants import TIME_SIGNATURE_KEY, NOTE_DENSITY_KEY, DEFAULT_NOTE_DENSITY_BINS, DEFAULT_POS_PER_QUARTER, BAR_KEY, \
    BOS_TOKEN, EOS_TOKEN
from representation import remi2midi
from tokens import DescriptionVocab, RemiVocab

desc_vocab = DescriptionVocab()
remi_vocab = RemiVocab()

dotenv_path = './variables.env'
load_dotenv(dotenv_path=dotenv_path)


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


class hide_prints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def generate_sample_from_description(description, model):
    # parse the given description to model input
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

    with hide_prints():
        # generate a sample based on the description
        sample = model.sample(batch, max_bars=n_bars, max_length=int(approx_n_tokens * 1.2))

    # convert the generated tokens to MIDI and write it to disk
    remi_events = remi_vocab.decode(sample["sequences"][0])
    pm = remi2midi(remi_events)

    iterator = os.getenv('ITERATOR', '-1')
    mid_path = f"./results/mid/sample_{iterator}.mid"
    pm.write(mid_path)

    # synthesize the generated MIDI and display it:
    audio = pm.fluidsynth()
    audio_path = f"./results/wav/sample{iterator}.wav"
    iterator_incremented = str(int(iterator) + 1)
    set_key(dotenv_path=dotenv_path, key_to_set='ITERATOR', value_to_set=iterator_incremented)

    soundfile.write(audio_path, audio, 44100)
    return IPython.display.Audio(audio_path)
