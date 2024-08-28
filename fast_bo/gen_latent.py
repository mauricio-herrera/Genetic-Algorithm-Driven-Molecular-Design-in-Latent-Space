import sys
sys.path.append('../')
import torch
import torch.nn as nn
from optparse import OptionParser
from tqdm import tqdm
import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import MolFromSmiles, MolToSmiles
from rdkit.Chem import rdmolops
import numpy as np
from fast_jtnn import *
from fast_jtnn import sascorer
import networkx as nx
import os

# Definir scorer
def scorer(smiles):
    smiles_rdkit = []
    for i in range(len(smiles)):
        try:
            mol = MolFromSmiles(smiles[i], sanitize=False)  # Desactivar sanitización
            if mol:
                Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)  # Sanitizar sin kekulización
                smiles_rdkit.append(MolToSmiles(mol, isomericSmiles=True))
            else:
                print(f"Invalid SMILES (null molecule): {smiles[i]}")
                smiles_rdkit.append(None)
        except Exception as e:
            print(f"Error processing SMILES {smiles[i]}: {e}")
            smiles_rdkit.append(None)

    logP_values = []
    SA_scores = []
    cycle_scores = []
    
    for smile in smiles_rdkit:
        if smile is None:
            logP_values.append(None)
            SA_scores.append(None)
            cycle_scores.append(None)
            continue
        
        try:
            mol = MolFromSmiles(smile)  # Reconvertir SMILES ya sanitizados (ahora debería ser seguro)
            logP_values.append(Descriptors.MolLogP(mol))
            SA_scores.append(-sascorer.calculateScore(mol))
            
            cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(mol)))
            if len(cycle_list) == 0:
                cycle_length = 0
            else:
                cycle_length = max([len(j) for j in cycle_list])
            if cycle_length <= 6:
                cycle_length = 0
            else:
                cycle_length = cycle_length - 6
            cycle_scores.append(-cycle_length)
        except Exception as e:
            print(f"Error calculating properties for SMILES {smile}: {e}")
            logP_values.append(None)
            SA_scores.append(None)
            cycle_scores.append(None)

    # Filtrar None antes de normalizar
    valid_indices = [i for i in range(len(logP_values)) if logP_values[i] is not None]

    SA_scores_filtered = np.array([SA_scores[i] for i in valid_indices])
    logP_values_filtered = np.array([logP_values[i] for i in valid_indices])
    cycle_scores_filtered = np.array([cycle_scores[i] for i in valid_indices])

    SA_scores_normalized = (SA_scores_filtered - np.mean(SA_scores_filtered)) / np.std(SA_scores_filtered)
    logP_values_normalized = (logP_values_filtered - np.mean(logP_values_filtered)) / np.std(logP_values_filtered)
    cycle_scores_normalized = (cycle_scores_filtered - np.mean(cycle_scores_filtered)) / np.std(cycle_scores_filtered)

    targets = (SA_scores_normalized + logP_values_normalized + cycle_scores_normalized)

    return (SA_scores, logP_values, cycle_scores, targets)

# Definir la función principal
def main_gen_latent(data_path, vocab_path, model_path, output_path='./', hidden_size=450, latent_size=56, depthT=20, depthG=3, batch_size=100):
    with open(data_path) as f:
        smiles = f.readlines()
    
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    for i in range(len(smiles)):
        smiles[i] = smiles[i].strip()

    vocab = [x.strip("\r\n ") for x in open(vocab_path)]
    vocab = Vocab(vocab)

    model = JTNNVAE(vocab, hidden_size, latent_size, depthT, depthG)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    latent_points = []
    for i in tqdm(range(0, len(smiles), batch_size)):
        batch = smiles[i:i + batch_size]
        try:
            mol_vec = model.encode_from_smiles(batch)
            latent_points.append(mol_vec.data.cpu().numpy())
        except Exception as e:
            print(f"Error processing batch: {e}")

    if latent_points:
        latent_points = np.vstack(latent_points)
        np.savetxt(os.path.join(output_path, 'latent_features.txt'), latent_points)

        # Llamar a scorer
        SA_scores, logP_values, cycle_scores, targets = scorer(smiles)
        np.savetxt(os.path.join(output_path, 'targets.txt'), targets)
        np.savetxt(os.path.join(output_path, 'logP_values.txt'), np.array(logP_values))
        np.savetxt(os.path.join(output_path, 'SA_scores.txt'), np.array(SA_scores))
        np.savetxt(os.path.join(output_path, 'cycle_scores.txt'), np.array(cycle_scores))
    else:
        print("No valid data to process.")

# Ejecutar el script
if __name__ == '__main__':
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = OptionParser()
    parser.add_option("-a", "--data", dest="data_path")
    parser.add_option("-v", "--vocab", dest="vocab_path")
    parser.add_option("-m", "--model", dest="model_path")
    parser.add_option("-o", "--output", dest="output_path", default='./')
    parser.add_option("-w", "--hidden", dest="hidden_size", default=450)
    parser.add_option("-l", "--latent", dest="latent_size", default=56)
    parser.add_option("-t", "--depthT", dest="depthT", default=20)
    parser.add_option("-g", "--depthG", dest="depthG", default=3)

    opts, args = parser.parse_args()

    hidden_size = int(opts.hidden_size)
    latent_size = int(opts.latent_size)
    depthT = int(opts.depthT)
    depthG = int(opts.depthG)

    main_gen_latent(opts.data_path, opts.vocab_path, opts.model_path, output_path=opts.output_path, hidden_size=hidden_size, latent_size=latent_size, depthT=depthT, depthG=depthG)

