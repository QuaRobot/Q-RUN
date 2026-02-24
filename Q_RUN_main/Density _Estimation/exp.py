import math
import time
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from neuralop import TFNO1d
from scipy.stats import norm

num_components = 25
num_samples = 1024
seed = 4230
np.random.seed(seed)
torch.manual_seed(seed)

mu = np.random.uniform(-8, 8, size=num_components)
sigma = np.random.uniform(0.08, 0.1, size=num_components)
weights = np.ones(num_components) / num_components
samples_per_component = (weights * num_samples).astype(int)

samples = []
for mean, std, count in zip(mu, sigma, samples_per_component):
    samples.append(np.random.normal(loc=mean, scale=std, size=count))
samples = np.concatenate(samples)
samples_tensor = torch.tensor(samples, dtype=torch.float32).view(-1, 1)

x_test = torch.linspace(-10, 10, 256, dtype=torch.float32).view(-1, 1)

test_input_full = torch.linspace(-10, 10, 10000, dtype=torch.float32).view(-1, 1)




class KANLinear(torch.nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            enable_standalone_scale_spline=True,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                    torch.arange(-spline_order, grid_size + spline_order + 1) * h
                    + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )

        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                    (
                            torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                            - 1 / 2
                    )
                    * self.scale_noise
                    / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order: -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                            (x - grid[:, : -(k + 1)])
                            / (grid[:, k:-1] - grid[:, : -(k + 1)])
                            * bases[:, :, :-1]
                    ) + (
                            (grid[:, k + 1:] - x)
                            / (grid[:, k + 1:] - grid[:, 1:(-k)])
                            * bases[:, :, 1:]
                    )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output

        output = output.reshape(*original_shape[:-1], self.out_features)
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
                torch.arange(
                    self.grid_size + 1, dtype=torch.float32, device=x.device
                ).unsqueeze(1)
                * uniform_step
                + x_sorted[0]
                - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
                regularize_activation * regularization_loss_activation
                + regularize_entropy * regularization_loss_entropy
        )


class KAN(nn.Module):
    def __init__(
            self,
            input_dim=1,
            output_dim=1,
            hidden_dim=16,
            num_layers=3,
            grid_size=3,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        layers_hidden = [input_dim] + [hidden_dim] * num_layers + [output_dim]

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )
        self.softplus = nn.Softplus()
    def forward(self, x: torch.Tensor, update_grid=True):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return self.softplus(x)

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )



def kl_divergence(p, q, x):
    epsilon = 1e-10
    p = np.clip(p, epsilon, None)
    q = np.clip(q, epsilon, None)
    return np.sum(p * np.log(p / q)) * np.diff(x)[0]


class MLP_Tanh(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softplus(),
        )

    def forward(self, x):
        return self.net(x)


class MLP_ReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus(),
        )

    def forward(self, x):
        return self.net(x)


class MLP_Fourier(nn.Module):
    def __init__(self):
        super().__init__()
        # 这里用一个简化的Fourier特征映射代替
        self.frequency_matrix = torch.randn(32, 1).cuda() # 32 frequencies
        self.linear1 = nn.Linear(64, 64)
        self.linear2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, x):
        x_proj = 2 * math.pi * x @ self.frequency_matrix.T
        fourier_feats = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        x = self.relu(self.linear1(fourier_feats))
        x = self.linear2(x)
        return self.softplus(x)


# Q_RUN 和 PWLNN 你之前给的也可以照搬，我这里直接示例：

class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        return self.fc3(x)


class Q_RUNLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim // 2)
        self.u_proj = SimpleMLP(8, 32, 2)
        self.scales = nn.Parameter(torch.randn(4))

    def forward(self, x):
        x = self.input_proj(x)
        out_list = []
        for i in range(4):
            scaled_x = x * self.scales[i]
            out_list.append(torch.sin(scaled_x))
            out_list.append(torch.cos(scaled_x))
        out = torch.stack(out_list, dim=-1)
        out = self.u_proj(out)
        return out.flatten(start_dim=-2)


class Q_RUN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            Q_RUNLayer(1, 32),
            Q_RUNLayer(32, 32),
            nn.Linear(32, 1),
            nn.Softplus(),
        )

    def forward(self, x):
        x = self.net[0](x)
        x = torch.sin(x)
        x = self.net[1](x)
        x = torch.sin(x)
        x = self.net[2](x)
        return self.net[3](x)


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
    def __init__(self, input_dim=1, output_dim=1, hidden_dim=64, num_layers=2, segments=4):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), PiecewiseLinear(segments)]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), PiecewiseLinear(segments)]
        layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(nn.Softplus())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class FNO(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, hidden_dim=64, num_layers=3):
        super().__init__()
        self.linear1 = nn.Linear(input_dim,hidden_dim)
        self.linear2 = nn.Linear(hidden_dim,output_dim)
        self.activate = nn.Softplus()

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
    def forward(self, output):
        output = self.linear1(output)
        output = self.net(output.unsqueeze(1))
        output = self.linear2(output).squeeze(1)
        return self.activate(output)


class FANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True, with_gate=False):
        super(FANLayer, self).__init__()
        self.input_linear_p = nn.Linear(input_dim, output_dim // 4, bias=bias)
        self.input_linear_g = nn.Linear(input_dim, (output_dim - output_dim // 2))
        self.activation = nn.GELU()
        if with_gate:
            self.gate = nn.Parameter(torch.randn(1, dtype=torch.float32))

    def forward(self, src):
        g = self.activation(self.input_linear_g(src))
        p = self.input_linear_p(src)

        if not hasattr(self, 'gate'):
            output = torch.cat((torch.cos(p), torch.sin(p), g), dim=-1)
        else:
            gate = torch.sigmoid(self.gate)
            output = torch.cat((gate * torch.cos(p), gate * torch.sin(p), (1 - gate) * g), dim=-1)
        return output


class SIREN(nn.Module):
        def __init__(self):
            super(SIREN, self).__init__()

            self.net = nn.Sequential(
                nn.Linear(1, 64),
                nn.Linear(64, 64),
                nn.Linear(64, 1),
                nn.Softplus()
            )

        def forward(self, x):
            x = self.net[0](x)
            x = torch.sin(x)
            x = self.net[1](x)
            x = torch.sin(x)
            x = self.net[2](x)
            x = self.net[3](x)
            return x

class FAN(nn.Module):
    def __init__(self):
        super(FAN, self).__init__()

        self.net = nn.Sequential(
            FANLayer(1, 72),
            FANLayer(72, 72),
            nn.Linear(72, 1),
            nn.Softplus()
        )

    def forward(self, x):
        x = self.net(x)
        return x


def test_model_timing(model_class, device, epochs=3000):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = model_class().to(device)

    input_data = samples_tensor.to(device)
    x_test_device = x_test.to(device)
    test_input_device = test_input_full.to(device)

    if model_class == KAN:
        optimizer = optim.LBFGS(model.parameters(), lr=0.1, max_iter=20, history_size=10)
        patience = 100
        counter = 0
        best_val_loss = float('inf')
        start_train = time.time()
        for epoch in range(epochs):
            def closure():
                optimizer.zero_grad()
                density_pred = model(input_data)
                density_test = model(x_test_device)
                integral_approx = torch.trapz(density_test.squeeze(), x_test_device.view(-1))
                density_pred_normalized = density_pred / (integral_approx + 1e-6)
                loss = -torch.mean(torch.log(density_pred_normalized + 1e-8))
                loss.backward()
                return loss

            loss = optimizer.step(closure)

            if epoch % 10 == 0:
                val_loss = loss.item()
                with torch.no_grad():
                    density_pred_test = model(x_test_device).cpu().numpy().flatten()
                    integral_approx_test = np.trapz(density_pred_test, x_test_device.cpu().numpy().flatten())
                    density_pred_test /= integral_approx_test

                    x_np = x_test_device.cpu().numpy().flatten()
                    true_density_test = np.zeros_like(x_np)
                    for i in range(num_components):
                        true_density_test += weights[i] * norm.pdf(x_np, mu[i], sigma[i])
                    true_density_test /= np.trapz(true_density_test, x_np)

                    kl_div = kl_divergence(true_density_test, density_pred_test, x_np)
                    print(f"Epoch {epoch}, Loss: {val_loss:.6f}, KL Divergence: {kl_div:.6f}")

                # early stopping
                if val_loss < best_val_loss - 1e-5:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), "best_kan_model.pt")
                    counter = 0
                    print(f"✅ Best model updated at epoch {epoch} (loss={val_loss:.6f})")
                else:
                    counter += 1
                    if counter >= patience:
                        print(f"🛑 Early stopping at epoch {epoch} (loss={val_loss:.6f})")
                        break
        train_time = time.time() - start_train

        with torch.no_grad():
            start_test = time.time()
            _ = model(test_input_device)
            test_time = time.time() - start_test

    else:
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        start_train = time.time()
        for _ in range(epochs):
            optimizer.zero_grad()
            pred = model(input_data)
            pred_test = model(x_test_device)
            integral = torch.trapz(pred_test.squeeze(), x_test_device.view(-1))
            pred_normalized = pred / (integral + 1e-6)
            loss = -torch.mean(torch.log(pred_normalized + 1e-8))
            loss.backward()
            optimizer.step()
        train_time = time.time() - start_train

        with torch.no_grad():
            start_test = time.time()
            _ = model(test_input_device)
            test_time = time.time() - start_test

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return train_time, test_time, num_params


# 主流程示例
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_classes = {
    "KAN": KAN,
    "MLP_Tanh": MLP_Tanh,
    "MLP_ReLU": MLP_ReLU,
    "MLP_Fourier": MLP_Fourier,
    "Q_RUN": Q_RUN,
    "PWLNN": PWLNN,
    "FAN": FAN,
    "FNO": FNO
}

results = {}
for name, cls in model_classes.items():
    print(f"Testing model {name} ...")
    try:
        train_t, test_t, n_params = test_model_timing(cls, device)
        results[name] = {
            "train_time": train_t,
            "test_time": test_t,
            "num_params": n_params,
        }
        print(f"{name}: train {train_t:.2f}s, test {test_t:.4f}s, params {n_params}")
    except Exception as e:
        print(f"Error testing {name}: {e}")

print("\nAll results:")
for name, res in results.items():
    print(f"{name}: train {res['train_time']:.2f}s | test {res['test_time']:.4f}s | params {res['num_params']}")

