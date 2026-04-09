import torch
from torch_geometric.data import Data
from rdkit import Chem
from utils.logger import logger


class PolymerGraphBuilder:
    def __init__(self):
        # List of atoms we expect to find in polymers
        self.allowed_atoms = ['C', 'O', 'N', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'Si', 'B', 'H']

    def _get_node_features(self, atom):
        """Extract physical features for each ATOM (Node)"""
        atom_type = atom.GetSymbol()
        type_idx = self.allowed_atoms.index(atom_type) if atom_type in self.allowed_atoms else len(self.allowed_atoms)

        features = [
            type_idx,  # Atom type (as index)
            atom.GetDegree(),  # The number of bonds in an atom
            atom.GetFormalCharge(),  # Charge
            int(atom.GetIsAromatic()),  # Aromaticity (1 or 0)
            atom.GetMass()  # Atomic mass (important for phonons!)
        ]
        return features

    def _get_edge_features(self, bond):
        """Extract features for each CONNECTION (Edge)"""
        features = [
            bond.GetBondTypeAsDouble(),  # Bond type (1.0 - single, 2.0 - double, 1.5 - aromatic)
            int(bond.IsInRing())  # Is the connection in a ring?
        ]
        return features

    def smiles_to_graph(self, smiles: str):
        """Turns SMILES into a Tensor Graph PyTorch Geometric"""
        clean_smiles = str(smiles).replace('[*]', 'C').replace('*', 'C')
        mol = Chem.MolFromSmiles(clean_smiles)

        if mol is None:
            return None

        # 1. Collect the features of NODES (matrix X)
        node_feats = [self._get_node_features(atom) for atom in mol.GetAtoms()]
        x = torch.tensor(node_feats, dtype=torch.float)

        # 2. Collect indices and features of EDGES (Connectionей)
        edge_indices = []
        edge_feats = []

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            # The graph is undirected: heat flows in both directions, so we add back-and-forth connections.
            edge_indices += [[i, j], [j, i]]

            feat = self._get_edge_features(bond)
            edge_feats += [feat, feat]

        if len(edge_indices) > 0:
            # edge_index must be in the format [2, num_edges]
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_feats, dtype=torch.float)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 2), dtype=torch.float)

        # 3. Pack everything into a Data object that is understandable for PyTorch Geometric
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return data

    def build_dataset(self, df, smiles_col='smiles', target_col='Tc'):
        """Converts the entire DataFrame into a list of graphs"""
        logger.info(f"Convert {len(df)} SMILES to graph PyTorch...")
        graph_list = []
        valid_indices = []

        for idx, row in df.iterrows():
            graph = self.smiles_to_graph(row[smiles_col])
            if graph is not None:
                # Add the target variable (Y) that the neural network will have to predict
                graph.y = torch.tensor([row[target_col]], dtype=torch.float)
                graph_list.append(graph)
                valid_indices.append(idx)

        logger.info(f"Successfully created {len(graph_list)} graphs.")
        return graph_list, df.loc[valid_indices]