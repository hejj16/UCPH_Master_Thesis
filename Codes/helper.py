import torch
import numpy as np
import itertools
import os
from Bio.Phylo.TreeConstruction import DistanceCalculator
from Bio import AlignIO
from Bio import Phylo
from Bio.Phylo.TreeConstruction import DistanceTreeConstructor, _DistanceMatrix


def load_blosum(BLOSUM_path="BLOSUM62.txt", n_c=21):
    # Load the BLOSUM matrix
    b = np.loadtxt(BLOSUM_path, dtype=str)
    B = b[list(range(1, n_c)) + [-1], :][:, list(range(1, n_c)) + [-1]].astype(float)
    B = torch.from_numpy(B)
    return B.float()


def get_AA_dict(BLOSUM_path="BLOSUM62.txt", n_c=21):
    # Encode amino acid
    b = np.loadtxt(BLOSUM_path, dtype=str)
    AA_dict = {symbol: index for index, symbol in enumerate(b[0, 1:n_c])}
    AA_dict["-"] = n_c - 1
    AA_dict["*"] = n_c - 1
    AA_dict["."] = n_c - 1

    back_AA_dict = {}
    for i in AA_dict.keys():
        back_AA_dict[AA_dict[i]] = i
    back_AA_dict[n_c-1] = "-"

    return AA_dict, back_AA_dict


def load_Unaligned_Data(MSA_path, device, AA_dict):
    # Load the file
    with open(MSA_path, "r") as f:
        f = f.readlines()
    msa = []
    names = []
    lengths = []
    for i in f:
        if i[0] != ">":
            msa_append = list(filter(None, [j if j not in ["*", ".", "-"] else None for j in i.strip()]))
            lengths.append(len(msa_append))
            msa.append(msa_append)
        else:
            names.append(i[1:].strip())
    N_L = max(lengths)
    for i in range(len(msa)):
        msa[i] += (N_L - len(msa[i])) * ["-"]
    msa_tensor = torch.tensor([list(map(lambda x: AA_dict[x], i)) for i in msa]).long()
    msa_char = ["".join(list(filter(lambda x: x != "-", i))) for i in msa]

    return msa_tensor.to(device), names, lengths, msa_char


def load_MSA(MSA_path, device, AA_dict):
    # Load the file
    with open(MSA_path, "r") as f:
        f = f.readlines()
    msa = []
    names = []
    for i in f:
        if i[0] != ">":
            msa_append = [j if j not in ["*", ".", "-"] else "-" for j in i.strip()]
            msa.append(msa_append)
        else:
            names.append(i[1:].strip())
    N_L = len(msa[0])
    msa_tensor = torch.tensor([list(map(lambda x: AA_dict[x], i)) for i in msa]).long()

    return msa_tensor.to(device)


def load_MSA_Data(MSA_path, device, AA_dict):
    # Load the file
    with open(MSA_path, "r") as f:
        f = f.readlines()
    msa = []
    names = []
    for i in f:
        if i[0] != ">":
            msa_append = [j if j not in ["*", ".", "-"] else "-" for j in i.strip()]
            msa.append(msa_append)
        else:
            names.append(i[1:].strip())
    msa_tensor = torch.tensor([list(map(lambda x: AA_dict[x], i)) for i in msa]).long()
    msa_char = ["".join(i) for i in msa]

    return msa_tensor.to(device), names, msa_char


# def get_tree_distance(input_sequences, tree_method="nj"):
#     if not os.path.exists("muscle3.8.31_i86linux64"):
#         os.system("wget https://drive5.com/muscle/downloads3.8.31/muscle3.8.31_i86linux64.tar.gz")
#         os.system("tar -xf muscle3.8.31_i86linux64.tar.gz")
#         os.system("rm muscle3.8.31_i86linux64.tar.gz")
#
#     with open("seqs.fa", "w") as f:
#         fastq_file = []
#         for idx, i in enumerate(input_sequences):
#             fastq_file.append(">%d\n" % idx)
#             fastq_file.append(i + "\n")
#         f.writelines(fastq_file)
#
#     os.system("./muscle3.8.31_i86linux64 -in seqs.fa -out seqs.afa")
#     os.system("rm seqs.fa")
#
#     aln = AlignIO.read('seqs.afa', "fasta")
#     calculator = DistanceCalculator('identity')
#     constructor = DistanceTreeConstructor(calculator, tree_method)
#     t = constructor.build_tree(aln)
#
#     mat = np.zeros([len(t.get_terminals()), len(t.get_terminals())])
#     for x, y in itertools.combinations(t.get_terminals(), 2):
#         v = t.distance(x, y)
#         mat[int(x.name)][int(y.name)] = v
#         mat[int(y.name)][int(x.name)] = v
#
#     os.system("rm seqs.afa, seqs.phy, muscle3.8.31_i86linux64.tar.gz")
#     return mat


def get_true_tree_distances(true_tree_path, S_names):
    t = Phylo.read(true_tree_path, "newick")
    mat = np.zeros([len(t.get_terminals()), len(t.get_terminals())])
    for x, y in itertools.combinations(t.get_terminals(), 2):
        v = t.distance(x, y)
        idx, idy = S_names.index(x.name), S_names.index(y.name)
        mat[idx][idy] = v
        mat[idy][idx] = v
    return mat


def build_tree(dm, names):
    d = _DistanceMatrix(names, [[j for j in i[:idx+1]] for idx, i in enumerate(dm)])
    constructor = DistanceTreeConstructor()
    tree = constructor.nj(d)
    return tree


def get_nj_tree_distance_including_ancestors(t, S_names):
    nodes = t.get_terminals() + t.get_nonterminals()
    mat = np.zeros([len(nodes), len(nodes)])
    ans_names = ["Inner" + str(i) for i in range(1, len(S_names)-1)]
    names = S_names + ans_names
    for x, y in itertools.combinations(nodes, 2):
        v = t.distance(x, y)
        x_name = x.name
        y_name = y.name
        idx, idy = names.index(x_name), names.index(y_name)
        mat[idx][idy] = v
        mat[idy][idx] = v
    return mat, ans_names






