import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool


class PolymerGNN(torch.nn.Module):
    def __init__(self, num_node_features=5, edge_dim=2, hidden_channels=128):
        super(PolymerGNN, self).__init__()
        torch.manual_seed(42)

        # GATv2Conv умеет "обращать внимание" на тип связи (edge_dim=2)
        # Увеличиваем "мозг" до 128 нейронов и добавляем 4 головы внимания (heads=4)
        self.conv1 = GATv2Conv(num_node_features, hidden_channels, heads=4, edge_dim=edge_dim, concat=False)
        self.conv2 = GATv2Conv(hidden_channels, hidden_channels, heads=4, edge_dim=edge_dim, concat=False)
        self.conv3 = GATv2Conv(hidden_channels, hidden_channels, heads=6, edge_dim=edge_dim, concat=False)

        # Полносвязные слои для финальной регрессии
        self.lin1 = Linear(hidden_channels * 2, hidden_channels)  # *2 потому что мы склеим 2 типа пулинга
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, x, edge_index, edge_attr, batch):
        # 1. Message Passing с учетом свойств химических связей (edge_attr)
        x = self.conv1(x, edge_index, edge_attr)
        x = F.elu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.elu(x)
        x = self.conv3(x, edge_index, edge_attr)
        x = F.elu(x)

        # 2. Слияние: берем и "среднюю" макромолекулу, и ее "максимальные" яркие черты
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)  # Склеиваем

        # 3. Регрессия с Dropout
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.lin1(x)
        x = F.elu(x)
        x = self.lin2(x)

        return x