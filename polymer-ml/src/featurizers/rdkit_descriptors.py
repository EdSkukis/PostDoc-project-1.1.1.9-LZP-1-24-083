from typing import List, Optional, Callable

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors

# Define a standard list of descriptor functions
DESCRIPTOR_FUNCTIONS: List[Callable[[Chem.Mol], float]] = [
    Descriptors.MolWt,
    Crippen.MolLogP,
    Descriptors.MolMR,
    Descriptors.TPSA,
    Lipinski.NumHDonors,
    Lipinski.NumHAcceptors,
    Lipinski.NumRotatableBonds,
    rdMolDescriptors.CalcFractionCSP3,
    Descriptors.RingCount,
    Lipinski.NumAromaticRings,
    rdMolDescriptors.CalcNumHeteroatoms,
    rdMolDescriptors.CalcHallKierAlpha,
    rdMolDescriptors.CalcKappa1,
    rdMolDescriptors.CalcKappa2,
    rdMolDescriptors.CalcKappa3,
    Descriptors.NumValenceElectrons,
]

# The length of the descriptor vector is now determined dynamically
NUM_DESCRIPTORS = len(DESCRIPTOR_FUNCTIONS)


def calc_descriptor_vector(mol: Optional[Chem.Mol]) -> np.ndarray:
    """
    Returns a descriptor vector for an RDKit molecule.
    If mol is None or an error occurs, it returns a zero vector of the correct length.

    The set of descriptors is chosen for QSPR:
      - size / mass
      - polarity / lipophilicity
      - flexibility
      - saturation
      - cyclicity / aromaticity
      - shape / topology
    """
    if mol is None:
        return np.zeros(NUM_DESCRIPTORS, dtype=np.float32)

    try:
        desc = [func(mol) for func in DESCRIPTOR_FUNCTIONS]
        desc_np = np.array(desc, dtype=np.float32)
        
        # Replace non-finite values (NaN, inf) with 0.0
        desc_np[~np.isfinite(desc_np)] = 0.0
        
        return desc_np

    except Exception:
        # In case of any other RDKit error, return a zero vector
        return np.zeros(NUM_DESCRIPTORS, dtype=np.float32)