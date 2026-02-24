import torch
import torch.nn as nn

class FANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True, with_gate = True):
        super(FANLayer, self).__init__()
        self.input_linear_p = nn.Linear(input_dim, output_dim//4, bias=bias)
        self.input_linear_g = nn.Linear(input_dim, (output_dim-output_dim//2))
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
            output = torch.cat((gate*torch.cos(p), gate*torch.sin(p), (1-gate)*g), dim=-1)
        return output



class Q_RUNLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim//2)
        self.u_proj = SimpleMLP(4,64,2)

    def forward(self, x):
        x = self.input_proj(x)
        cos_x = torch.cos(x)  # [B, H]
        sin_x = torch.sin(x)  # [B, H]

        out = torch.stack([sin_x,cos_x,sin_x,cos_x], dim=-1)
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








