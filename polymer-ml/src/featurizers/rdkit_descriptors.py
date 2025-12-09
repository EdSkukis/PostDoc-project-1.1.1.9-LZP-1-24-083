from typing import List, Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors


def calc_descriptor_vector(mol: Optional[Chem.Mol]) -> np.ndarray:
    """
    Возвращает вектор дескрипторов для молекулы RDKit.
    Если mol is None, возвращает вектор нулей той же длины.

    Набор дескрипторов подобран для QSPR:
      - размер / масса
      - полярность / липофильность
      - гибкость
      - насыщенность
      - цикличность / ароматичность
      - форма / топология
    """
    if mol is None:
        # ОБЯЗАТЕЛЬНО: длина этого списка должна совпадать с числом дескрипторов ниже
        return np.zeros(16, dtype=np.float32)

    try:
        # 1) базовые физико-химические свойства
        mw = Descriptors.MolWt(mol)                    # молекулярная масса
        logp = Crippen.MolLogP(mol)                    # логP (липофильность)
        mr = Descriptors.MolMR(mol)                    # молярная рефрактивность
        tpsa = Descriptors.TPSA(mol)                   # полярная поверхность

        # 2) HBD/HBA, гибкость
        hbd = Lipinski.NumHDonors(mol)                 # доноры H-связей
        hba = Lipinski.NumHAcceptors(mol)              # акцепторы H-связей
        rot_bonds = Lipinski.NumRotatableBonds(mol)    # вращаемые связи

        # 3) насыщенность / ароматичность / цикличность
        frac_csp3 = rdMolDescriptors.CalcFractionCSP3(mol)   # доля sp3-углеродов
        ring_count = Descriptors.RingCount(mol)              # все кольца
        arom_rings = Lipinski.NumAromaticRings(mol)          # ароматические кольца
        hetero_atoms = rdMolDescriptors.CalcNumHeteroatoms(mol)  # гетероатомы

        # 4) топологические дескрипторы Kier & Hall
        hall_kier_alpha = rdMolDescriptors.CalcHallKierAlpha(mol)
        kappa1 = rdMolDescriptors.CalcKappa1(mol)
        kappa2 = rdMolDescriptors.CalcKappa2(mol)
        kappa3 = rdMolDescriptors.CalcKappa3(mol)

        # 5) валентные электроны
        val_electrons = Descriptors.NumValenceElectrons(mol)

        desc = np.array(
            [
                mw,
                logp,
                mr,
                tpsa,
                hbd,
                hba,
                rot_bonds,
                frac_csp3,
                ring_count,
                arom_rings,
                hetero_atoms,
                hall_kier_alpha,
                kappa1,
                kappa2,
                kappa3,
                val_electrons,
            ],
            dtype=np.float32,
        )
        return desc

    except Exception:
        # На всякий случай: если RDKit где-то упал — вернём нули
        return np.zeros(16, dtype=np.float32)