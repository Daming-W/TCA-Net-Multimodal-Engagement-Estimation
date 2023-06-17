import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

class ModalityFusionModel(nn.Module):
    def __init__(self, input_shape, hidden_units, output_units):
        super(ModalityFusionModel, self).__init__()
        self.input_shape = input_shape
        self.hidden_units = hidden_units
        self.output_units = output_units

        self.fc_layer = nn.Linear(input_shape, hidden_units)
        self.fusion_layer = nn.Linear(hidden_units, hidden_units)
        self.output_layer = nn.Linear(hidden_units, output_units)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        fused_output = self.relu(self.fc_layer(inputs))
        fused_output = self.relu(self.fusion_layer(fused_output))
        output = self.output_layer(fused_output)

        return self.relu(output)



if __name__ == '__main__':
    # Example usage
    input_shapes = 83  # Example input shapes for 3 modalities
    hidden_units = 64
    output_units = 1

    model = ModalityFusionModel(input_shapes, hidden_units, output_units)
    # Example forward pass
    input_data = torch.randn(161798, 83)  # Your concatenated input data
    output = model(input_data)

    print(output)