"""
    These files are heavily based on the figaro jupyter notebook online demonstration
"""""

from descriptions import my_simple_description, simple_description, chords_description

import torch
import generate
from model import Seq2SeqModule


def load_model():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = Seq2SeqModule.load_from_checkpoint("checkpoints/figaro-expert.ckpt")
    model.freeze()
    model.eval()
    model.to(device)

    return model


if __name__ == '__main__':
    print("started")
    model = load_model()

    # generate.generate_sample_from_description(simple_description, model)
    generate.generate_sample_from_description(my_simple_description, model)
    # generate.generate_sample_from_description(chords_description, model)