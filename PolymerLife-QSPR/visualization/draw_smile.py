from rdkit import Chem
from rdkit.Chem import Draw

smiles = "*C(=O)Oc1ccc(OC(=O)c2ccc3c(c2)C(=O)N(c2ccc(-c4ccc(N5C(=O)c6ccc(*)cc6C5=O)cc4C(F)(F)F)c(C(F)(F)F)c2)C3=O)cc1"

# Заменяем '*' на 'I' (йод) для визуализации точек соединения цепей
mol = Chem.MolFromSmiles(smiles.replace('*', 'I'))

if mol:
    img = Draw.MolToImage(mol, size=(800, 600))
    img.save("fluorinated_polyimide.png")
    print("Изображение успешно сохранено как fluorinated_polyimide.png")
else:
    print("Ошибка: Не удалось распознать SMILES.")