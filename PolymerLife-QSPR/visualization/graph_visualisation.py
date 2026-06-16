import networkx as nx
import matplotlib.pyplot as plt
from rdkit import Chem

def build_polymer_graph(smiles):
    # Заменяем звёздочки [*] на фиктивный атом (например, Йод 'I'),
    # чтобы RDKit корректно считал структуру
    mol = Chem.MolFromSmiles(smiles.replace('*', 'I'))
    if not mol:
        return None

    G = nx.Graph()

    # Добавляем узлы (атомы) с атрибутами
    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(),
                   symbol=atom.GetSymbol())

    # Добавляем ребра (химические связи)
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(),
                   bond.GetEndAtomIdx(),
                   bond_type=str(bond.GetBondType()))
    return G

# Твоя структура
smiles = "*C(=O)Oc1ccc(OC(=O)c2ccc3c(c2)C(=O)N(c2ccc(-c4ccc(N5C(=O)c6ccc(*)cc6C5=O)cc4C(F)(F)F)c(C(F)(F)F)c2)C3=O)cc1"
G = build_polymer_graph(smiles)

# Визуализация
if G:
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, k=0.5, seed=42) # Алгоритм распределения узлов

    # Раскрасим разные атомы для наглядности
    labels = {n: f"{n}:{d['symbol']}" for n, d in G.nodes(data=True)}
    color_map = {'C': 'lightgray', 'O': 'tomato', 'N': 'skyblue', 'F': 'lightgreen', 'I': 'gold'}
    node_colors = [color_map.get(G.nodes[n]['symbol'], 'cyan') for n in G]

    nx.draw(G, pos, labels=labels, with_labels=True,
            node_color=node_colors, node_size=800,
            font_size=8, edge_color='gray', alpha=0.9)

    plt.title("Topological Graph Representation (GNN Input)")
    plt.show()