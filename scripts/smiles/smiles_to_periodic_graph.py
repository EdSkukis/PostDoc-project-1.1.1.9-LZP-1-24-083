
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import networkx as nx
from rdkit import Chem

@dataclass
class ParsedMol:
    mol: Chem.Mol
    dummy_ids: List[int]

def _normalize_bigsmiles(s: str) -> str:
    """
    Minimal normalization:
      - Strip outer braces {...} if present.
      - Keep attachment points [*] as-is.
      - This is NOT a full BigSMILES parser; it supports simple repeat units.
    """
    s = s.strip()
    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1].strip()
    return s

def parse_smiles_or_bigsmiles(s: str) -> ParsedMol:
    """Parse SMILES or simple BigSMILES-like repeat unit with [*] endpoints."""
    s_norm = _normalize_bigsmiles(s)
    mol = Chem.MolFromSmiles(s_norm)
    if mol is None:
        raise ValueError(f"RDKit failed to parse: {s!r} -> {s_norm!r}")
    dummy_ids = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0]  # [*] atoms
    return ParsedMol(mol=mol, dummy_ids=dummy_ids)

def mol_to_networkx(mol: Chem.Mol) -> nx.Graph:
    """Convert RDKit Mol to a NetworkX undirected graph with atom/bond labels."""
    G = nx.Graph()
    for a in mol.GetAtoms():
        i = a.GetIdx()
        G.add_node(i, symbol=a.GetSymbol(), atomic_num=a.GetAtomicNum())
    for b in mol.GetBonds():
        i = b.GetBeginAtomIdx()
        j = b.GetEndAtomIdx()
        G.add_edge(i, j, order=int(b.GetBondTypeAsDouble()))
    return G

def add_periodic_edge_if_needed(G: nx.Graph, dummy_ids: List[int]) -> None:
    """If there are exactly two [*] dummy atoms, connect them with a labeled 'periodic' edge."""
    if len(dummy_ids) == 2:
        i, j = dummy_ids
        G.add_edge(i, j, order=0, periodic=True)
    # If more than two dummies exist, user can connect specific pairs externally.

def smiles_or_bigsmiles_to_graph(s: str) -> nx.Graph:
    """
    High-level function:
      - parses SMILES/BigSMILES (simple),
      - builds a NetworkX graph,
      - adds a 'periodic' edge between two [*] if present.
    """
    pm = parse_smiles_or_bigsmiles(s)
    G = mol_to_networkx(pm.mol)
    add_periodic_edge_if_needed(G, pm.dummy_ids)
    return G

def draw_graph(G: nx.Graph, title: str, out_png: str) -> str:
    """Draw NetworkX graph with atom symbols and bond orders; save to PNG."""
    import matplotlib.pyplot as plt
    pos = nx.spring_layout(G, seed=42)
    node_labels = {i: G.nodes[i].get("symbol", "?") for i in G.nodes}
    edge_labels = {(u, v): G.edges[u, v].get("order", "") for u, v in G.edges}
    plt.figure(figsize=(6,5))
    nx.draw(G, pos, with_labels=False, node_size=700)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=9)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()
    return out_png