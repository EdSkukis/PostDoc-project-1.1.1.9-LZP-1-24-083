import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from utils.logger import logger


class SafeDescriptorBuilder:
    def __init__(self):
        # Using Morgan Fingerprints (512 bits)
        self.mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=512)
        # Top 5 Fundamental Descriptors
        self.desc_funcs = {
            'MolWt': Descriptors.MolWt,
            'LogP': Descriptors.MolLogP,
            'NumHDonors': Descriptors.NumHDonors,
            'NumHAcceptors': Descriptors.NumHAcceptors,
            'TPSA': Descriptors.TPSA
        }

    def process_smiles(self, smiles: str):
        try:
            # Cleaning of polymer tags (BigSMILES)
            clean_smiles = str(smiles).replace('[*]', 'C').replace('*', 'C')
            mol = Chem.MolFromSmiles(clean_smiles)

            if mol is None:
                return None

            features = [func(mol) for func in self.desc_funcs.values()]
            fp = self.mfpgen.GetFingerprint(mol)

            return np.concatenate([features, np.array(fp)])

        except Exception as e:
            logger.debug(f"SMILES parsing error {smiles}: {str(e)}")
            return None

    def get_feature_names(self):
        return list(self.desc_funcs.keys()) + [f"fp_{i}" for i in range(512)]

    def featurize_dataframe(self, df: pd.DataFrame, smiles_col: str = 'smiles') -> pd.DataFrame:
        logger.info("Generation of molecular features...")
        features_list = []
        valid_indices = []

        for idx, row in df.iterrows():
            feats = self.process_smiles(row[smiles_col])
            if feats is not None:
                features_list.append(feats)
                valid_indices.append(idx)

        feat_df = pd.DataFrame(features_list, columns=self.get_feature_names(), index=valid_indices)
        logger.info(f"Successfully processed {len(feat_df)} из {len(df)} molecules.")

        return pd.concat([df.loc[valid_indices], feat_df], axis=1)