"""
pairwise distances:

1. reconstruction accuracy: average accuracy / average accuracy removing gaps in both sequences
2. ASR accuracy: accuracy after re-aligning

"""
import torch
import numpy as np
from Bio import Align


def reconstruction_accuracy(seqs1, seqs2, gap_index=23, remove_gap=True):
    """
    Calculate %identity between 2 groups of sequences
    :param seqs1: (N, L, n_c) tensor
    :param seqs2: (N, L, n_c) tensor
    :param gap_index: index of gap. default 23
    :param remove_gap: if removing the gaps when calculating %identity
    :return: (N, ) numpy array of %identity
    """

    seqs = torch.cat([seqs1.detach().cpu().argmax(dim=-1, keepdim=True),
                      seqs2.detach().cpu().argmax(dim=-1, keepdim=True)],
                     dim=-1).numpy()
    if remove_gap:
        indices = np.any(seqs != gap_index, axis=-1)
        accuracy = (seqs[:, :, 0] == seqs[:, :, 1])
        return np.sum(accuracy * indices, axis=1) / np.sum(indices, axis=1)
    else:
        accuracy = (seqs[:, :, 0] == seqs[:, :, 1])
        return np.mean(accuracy, axis=1)


def asr_accuracy(seqs_ture, seqs_predicted):
    """
    Calculate %identity between ground truth and predicted sequences after realignment
    :param seqs_ture: list of sequences (strings)
    :param seqs_predicted: list of sequences (strings)
    :return: (N, ) numpy array of %identity
    """

    aligner = Align.PairwiseAligner(match_score=1.0, mismatch_score=-1, gap_score=-1)

    def _score(i):
        if seqs_ture[i] == "" or seqs_predicted[i] == "":
            return 0
        align = aligner.align(seqs_ture[i], seqs_predicted[i])
        # return align.score / min(map(lambda x: len(x.__str__().split("\n")[0]), align[:100]))
        align = align[0].__str__().split("\n")[1]
        return align.count("|") / len(align)

    return np.array(list(map(_score, range(len(seqs_ture)))))


def tensor2char(seqs_predicted, back_AA_dict):

    f = lambda x: "".join(list(filter(lambda y: y != "-", map(lambda y: back_AA_dict[y], x))))
    return list(map(f, seqs_predicted.detach().cpu().argmax(dim=-1).numpy()))




