import argparse
import math
import numpy as np
import torch
import torch.nn as nn
from neuralop import TFNO1d
from torch.optim import AdamW, LBFGS
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

seed = 4230
np.random.seed(seed)
torch.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(dataset_name, batch_size):
    if dataset_name == "h2o":
        data = np.load("training-data/H2O_data.npy", allow_pickle=True).item()
        energy = np.array(data['Energy']).reshape(-1, 1)
        forces = data['Forces']
        positions = data['cc']
        shape = positions.shape
        scaler = MinMaxScaler(feature_range=(-1, 1))
        energy = scaler.fit_transform(energy)
        forces = forces * scaler.scale_

        data_pos = np.zeros((shape[0], 2, 3))
        data_pos[:, 0, :] = positions[:, 1, :] - positions[:, 0, :]
        data_pos[:, 1, :] = positions[:, 2, :] - positions[:, 0, :]
        positions = data_pos.copy()
        forces = forces[:, 1:, :]

    elif dataset_name == "two_h2o":
        data = np.load("training-data/two_H2O_data.npy", allow_pickle=True).item()
        energy = np.array(data["Energy"]).reshape(-1, 1)
        forces = data["Forces"]
        positions = data["cc"]
        scaler = MinMaxScaler(feature_range=(-1, 1))
        energy = scaler.fit_transform(energy)
        forces = forces * scaler.scale_
        O1 = positions[:, 1, :]
        O2 = positions[:, 4, :]
        center = (O1 + O2) / 2
        positions = positions - center[:, None, :]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    shape = positions.shape
    indices = np.arange(shape[0])
    np.random.shuffle(indices)
    split = int(0.8 * shape[0])
    train_idx, test_idx = indices[:split], indices[split:]

    train_dataset = TensorDataset(
        torch.tensor(positions[train_idx], dtype=torch.float32),
        torch.tensor(energy[train_idx], dtype=torch.float32),
        torch.tensor(forces[train_idx], dtype=torch.float32),
    )
    test_dataset = TensorDataset(
        torch.tensor(positions[test_idx], dtype=torch.float32),
        torch.tensor(energy[test_idx], dtype=torch.float32),
        torch.tensor(forces[test_idx], dtype=torch.float32),
    )
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(test_dataset, batch_size=batch_size),
        torch.tensor(positions[train_idx], dtype=torch.float32),
        torch.tensor(energy[train_idx], dtype=torch.float32),
        torch.tensor(forces[train_idx], dtype=torch.float32),
    )

class PiecewiseLinear(nn.Module):
    def __init__(self, segments=4):
        super().__init__()
        self.segments = segments
        self.breakpoints = nn.Parameter(torch.linspace(-1, 1, segments))
        self.slopes = nn.Parameter(torch.ones(segments))

    def forward(self, x):
        out = 0
        for i in range(self.segments):
            out += self.slopes[i] * torch.relu(x - self.breakpoints[i])
        return out

class PWLNN(nn.Module):
    def __init__(self, input_dim=18, output_dim=1, hidden_dim=64, num_layers=4, segments=4):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), PiecewiseLinear(segments)]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), PiecewiseLinear(segments)]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.layers = nn.Sequential(*layers)
        self.input_dim= input_dim
    def forward(self, x):
        x = x.view(-1, self.input_dim)
        for layer in self.layers:
            x = layer(x)
        return x

class FNO(nn.Module):
    def __init__(self, input_dim=6, output_dim=1, hidden_dim=64, num_layers=3):
        super().__init__()
        self.input_dim= input_dim
        self.linear1 = nn.Linear(input_dim,hidden_dim)
        self.linear2 = nn.Linear(hidden_dim,output_dim)

        self.net =  TFNO1d(
                in_channels=1,
                out_channels=1,
                hidden_channels= 10,
                n_modes_height= 2,
                depth=num_layers,
                mlp=True,
                lifting_channels=hidden_dim,
                projection_channels=hidden_dim
            )
    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = self.linear1(x)
        x = self.net(x.unsqueeze(1))
        x = self.linear2(x).squeeze(1)
        return x



class Sine(nn.Module):
    def forward(self, input):
        return torch.sin(input)

class SIREN(nn.Module):
    def __init__(self, input_dim=6, hidden_features=35, hidden_layers=3, out_features=1):

        super().__init__()
        layers = []

        layers.append(nn.Linear(input_dim, hidden_features))
        for _ in range(hidden_layers):
            layers.append(Sine())
        layers.append(nn.Linear(hidden_features, out_features))
        self.net = nn.Sequential(*layers)
        self.input_dim= input_dim

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.net(x)


class FANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True):
        super(FANLayer, self).__init__()
        self.input_linear_p = nn.Linear(input_dim, output_dim // 4,
                                        bias=bias)  # There is almost no difference between bias and non-bias in our experiments.
        self.input_linear_g = nn.Linear(input_dim, (output_dim - output_dim // 2))
        self.activation = nn.GELU()

    def forward(self, src):
        g = self.activation(self.input_linear_g(src))
        p = self.input_linear_p(src)

        output = torch.cat((torch.cos(p), torch.sin(p), g), dim=-1)
        return output



class FAN(nn.Module):
    def __init__(self, input_dim=6, hidden_features=35, out_features=1):
        self.net = nn.Sequential(


            nn.Linear(input_dim,hidden_features),
            FANLayer(hidden_features, hidden_features),
            FANLayer(hidden_features, hidden_features),
            nn.Linear(hidden_features, out_features),

        )

        self.input_dim= input_dim

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = self.net(x)
        return x




class FourierFeatures(nn.Module):
    def __init__(self, in_channels, out_channels, learnable_features=False):
        super(FourierFeatures, self).__init__()
        frequency_matrix = torch.normal(mean=torch.zeros(out_channels, in_channels),
                                        std=1.0)
        if learnable_features:
            self.frequency_matrix = nn.Parameter(frequency_matrix)
        else:
            self.register_buffer('frequency_matrix', frequency_matrix)
        self.learnable_features = learnable_features
        self.num_frequencies = frequency_matrix.shape[0]
        self.coordinate_dim = frequency_matrix.shape[1]
        # Factor of 2 since we consider both a sine and cosine encoding
        self.feature_dim = 2 * self.num_frequencies

    def forward(self, coordinates):

        prefeatures = torch.einsum('oi,bli->blo', self.frequency_matrix.to(coordinates.device), coordinates.unsqueeze(1))
        cos_features = torch.cos(2 * math.pi * prefeatures)
        sin_features = torch.sin(2 * math.pi * prefeatures)
        return torch.cat((cos_features, sin_features), dim=2).squeeze()


class RFF(nn.Module):
    def __init__(self, input_dim=6, hidden_features=35,hidden_layers=2,out_features=1):

        super().__init__()
        layers = []
        layers.append(FourierFeatures(input_dim, hidden_features // 2))
        for _ in range(hidden_layers):
            layers.append(nn.Linear(hidden_features, hidden_features))
        layers.append(nn.Linear(hidden_features, out_features))
        self.net = nn.Sequential(*layers)
        self.input_dim= input_dim

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.net(x)




class Q_RUN(nn.Module):
    def __init__(self, input_dim=6, hidden_features=35, out_features=1):
        super().__init__()

        self.net = nn.Sequential(

            Q_RUN_layer(input_dim, hidden_features),
            Q_RUN_layer(hidden_features, hidden_features),
            nn.Linear(hidden_features, out_features),

        )

        self.input_dim= input_dim

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = self.net(x)
        return x


class Q_RUN_layer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim//2)
        self.u_proj = SimpleMLP(8,32,2)

    def forward(self, x):
        x = self.input_proj(x)
        cos_x = torch.cos(x)  # [B, H]
        sin_x = torch.sin(x)  # [B, H]

        out = torch.stack([sin_x,cos_x,sin_x,cos_x,sin_x,cos_x,sin_x,cos_x], dim=-1)
        out = self.u_proj(out)
        return out.flatten(start_dim=-2)


class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.tanh   = nn.Tanh()
    def forward(self, x):
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

def get_model(name, input_dim, output_dim):
    name = name.lower()
    if name == "pwlnn":
        return PWLNN(input_dim=input_dim, output_dim=output_dim).to(device)
    elif name == "siren":
        return SIREN(input_dim=input_dim, out_features=output_dim).to(device)
    elif name == "fno":
        return FNO(input_dim=input_dim, output_dim=output_dim).to(device)
    elif name == "fan":
        return FAN(input_dim=input_dim, out_features=output_dim).to(device)
    elif name == "rff":
        return RFF(input_dim=input_dim, out_features=output_dim).to(device)
    elif name == "qrun":
        return Q_RUN(input_dim=input_dim, out_features=output_dim).to(device)
    else:
        raise ValueError(f"Unknown model: {name}")

def train(model, train_loader, test_loader, positions_train, energy_train, forces_train, args):
    mse_loss = nn.MSELoss()
    force_loss_weight = 100
    full_train_pos = positions_train.to(device).requires_grad_(True)
    full_train_energy = energy_train.to(device)
    full_train_forces = forces_train.to(device)

    optimizer1 = AdamW(model.parameters(), lr=args.lr)
    optimizer2 = LBFGS(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.1,
        history_size=40,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
    )

    print("Start AdamW pretraining...")
    for epoch in tqdm(range(args.epochs)):
        optimizer1.zero_grad()
        pred_energy = model(full_train_pos)
        grad_outputs = torch.ones_like(pred_energy)
        pred_forces = -torch.autograd.grad(
            outputs=pred_energy,
            inputs=full_train_pos,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
        )[0]
        loss_energy = mse_loss(pred_energy, full_train_energy)
        loss_force = mse_loss(pred_forces, full_train_forces)
        loss = loss_energy + force_loss_weight * loss_force
        loss.backward()
        optimizer1.step()

        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                test_energy = []
                test_true = []
                for pos_batch, energy_batch, _ in test_loader:
                    pos_batch = pos_batch.to(device)
                    pred = model(pos_batch)
                    test_energy.append(pred.cpu())
                    test_true.append(energy_batch)
                test_energy = torch.cat(test_energy).numpy()
                test_true = torch.cat(test_true).numpy()
                test_mse = ((test_energy - test_true) ** 2).mean()
            print(f"Epoch {epoch+1} - Train Energy Loss: {loss_energy.item():.6f}, "
                  f"Force Loss: {loss_force.item():.6f}, Test MSE: {test_mse:.6f}")
            model.train()

    def closure():
        optimizer2.zero_grad()
        pred_energy = model(full_train_pos)
        grad_outputs = torch.ones_like(pred_energy)
        pred_forces = -torch.autograd.grad(
            outputs=pred_energy,
            inputs=full_train_pos,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
        )[0]
        loss_energy = mse_loss(pred_energy, full_train_energy)
        loss_force = mse_loss(pred_forces, full_train_forces)
        loss = loss_energy + force_loss_weight * loss_force
        loss.backward()
        return loss

    print("Start LBFGS fine-tuning...")
    for epoch in tqdm(range(1000)):
        optimizer2.step(closure)
        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                pred_energy = model(full_train_pos)
                test_mse = ((pred_energy.cpu().detach().numpy() - energy_train.numpy()) ** 2).mean()
            print(f"LBFGS Epoch {epoch+1} - Train Energy Loss: {mse_loss(pred_energy, full_train_energy).item():.6f}, Test Energy MSE: {test_mse:.6f}")
            model.train()

# ----------------- 主程序 ----------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name: h2o or two_h2o")
    parser.add_argument("--model", type=str, required=True, help="Model name: pwlnn, siren, fno, fan, rff, qrun")
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    input_dim = 18 if args.dataset == "two_h2o" else 6  
    output_dim = 1

    train_loader, test_loader, pos, ene, frc = load_data(args.dataset, args.batch_size)
    model = get_model(args.model, input_dim=input_dim, output_dim=output_dim)
    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters")
    train(model, train_loader, test_loader, pos, ene, frc, args)

