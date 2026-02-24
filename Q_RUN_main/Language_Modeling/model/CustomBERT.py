import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertForSequenceClassification, BertConfig

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


class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class Q_RUNLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_reuploads=4):
        super().__init__()
        self.n_reuploads = n_reuploads
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim//2)
        self.u_proj = SimpleMLP(6,32,2)

    def forward(self, x):
        x = self.input_proj(x)
        cos_x = torch.cos(x)  # [B, H]
        sin_x = torch.sin(x)  # [B, H]

        out = torch.stack([sin_x,cos_x,sin_x,cos_x,sin_x,cos_x], dim=-1)

        out = self.u_proj(out)
        return out.flatten(start_dim=-2)



class CustomBertClassifier(BertForSequenceClassification):
    def __init__(self, num_labels=2, num_hidden_layers=12, replace_ffn=False, with_gate=False):
        config = BertConfig.from_pretrained("/mnt/fanmodel/bert", num_labels=num_labels)

        config.num_hidden_layers = num_hidden_layers
        super(CustomBertClassifier, self).__init__(config)
        if True: # replace the two linear layers in FFN for each layer
            # for layer in self.bert.encoder.layer:
            #     layer.intermediate = BertIntermediate_withFAN(config) # replace the intermediate layer because we don't need the activation function within the bert intermediate layer, which is already implemented in the FANLayer
            #     layer.output.dense = FANLayer(config.intermediate_size, config.hidden_size, with_gate=with_gate)
            for layer in self.bert.encoder.layer:
                layer.intermediate = BertIntermediate_withaQIFAN(config) # replace the intermediate layer because we don't need the activation function within the bert intermediate layer, which is already implemented in the FANLayer
                layer.output.dense = Q_RUNLayer(config.intermediate_size, config.hidden_size, 1)


class BertIntermediate_withaQIFAN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = Q_RUNLayer(config.hidden_size, config.intermediate_size,1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        return hidden_states



class BertIntermediate_withFAN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = FANLayer(config.hidden_size, config.intermediate_size, )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        return hidden_states
