import os
import torch
import pandas as pd
import numpy as np
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from utils.logger import logger
from utils.config import PIPELINE_CONFIG
from smiles.graph_builder import PolymerGraphBuilder
from models.gnn_model import PolymerGNN


def train_and_test_gnn():
    logger.info("=== ИЗОЛИРОВАННЫЙ ТЕСТ: Графовая Нейросеть (GNN) для Tc ===")
    torch.manual_seed(42)

    # 1. Загрузка данных
    data_path = "data/extended_polymer_dataset.csv"
    if not os.path.exists(data_path):
        data_path = "data/raw/extended_polymer_dataset.csv"

    df = pd.read_csv(data_path)
    x_col = PIPELINE_CONFIG["x"]["col_name"]
    y1_col = PIPELINE_CONFIG["y1"]["col_name"]
    df = df.dropna(subset=[x_col, y1_col])

    # 2. Превращаем SMILES в Графы PyTorch
    builder = PolymerGraphBuilder()
    graph_list, _ = builder.build_dataset(df, smiles_col=x_col, target_col=y1_col)

    # 3. Разбиваем на train/test (80/20)
    train_graphs, test_graphs = train_test_split(graph_list, test_size=0.2, random_state=42)

    # DataLoader сам упаковывает графы разного размера в удобные "батчи"
    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)

    # 4. Инициализация GAT-нейросети и умного оптимизатора
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PolymerGNN(hidden_channels=128).to(device)

    # Чуть меньший стартовый шаг, чтобы не перепрыгнуть оптимум
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    # Расписание: если Loss не падает 20 эпох, уменьшаем шаг в 2 раза
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
    criterion = torch.nn.MSELoss()

    logger.info("Начинаем обучение Attention GNN (400 эпох)...")

    # 5. Усиленный цикл обучения
    for epoch in range(1, 401):
        model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            # ПЕРЕДАЕМ EDGE_ATTR В МОДЕЛЬ!
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            loss = criterion(out.squeeze(), data.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        epoch_loss = total_loss / len(train_loader)
        scheduler.step(epoch_loss)  # Обновляем расписание

        if epoch % 50 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"Эпоха {epoch:03d} | Loss: {epoch_loss:.6f} | LR: {current_lr:.6f}")

    # 6. Оценка
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            y_pred.extend(out.squeeze().cpu().numpy().tolist())
            y_true.extend(data.y.cpu().numpy().tolist())

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    logger.info(f"=== Финальный результат Graph Attention ===")
    logger.info(f"R2 Score: {r2:.4f}")
    logger.info(f"MAE: {mae:.4f}")
    logger.info(f"RMSE: {rmse:.4f}")
if __name__ == "__main__":
    train_and_test_gnn()