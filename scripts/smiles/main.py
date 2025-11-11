from smiles_to_periodic_graph import smiles_or_bigsmiles_to_graph, draw_graph

G = smiles_or_bigsmiles_to_graph("{[*]OC(=O)NH[*]}")
draw_graph(G, "PU periodic repeat", "pu_periodic.png")