"""Export sNNake v5 direction model to ONNX for web inference."""
import sys
sys.path.insert(0, "src")

import torch
import torch.nn as nn
from snnake.model import StructuredWorldModel

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Load trained model
model = StructuredWorldModel().to(device)
state = torch.load("checkpoints_v5/best.pt", map_location=device, weights_only=True)
model.load_state_dict(state)
model.eval()
print(f"Model loaded ({model.get_num_params():,} params)")

# Direction MLP wrapper: (direction, action) → dir_logits (4-class)
class DirNetWrapper(nn.Module):
    def __init__(self, dir_net):
        super().__init__()
        self.dir_net = dir_net

    def forward(self, direction, action):
        x = torch.cat([direction, action], dim=1)
        return self.dir_net(x)

dir_model = DirNetWrapper(model.dir_net).to(device).eval()

# Export with fixed batch=1 (ONNX Runtime Web handles this fine)
dummy_dir = torch.randn(1, 4, device=device)
dummy_act = torch.randn(1, 3, device=device)

torch.onnx.export(
    dir_model,
    (dummy_dir, dummy_act),
    "web/direction_model.onnx",
    input_names=["direction", "action"],
    output_names=["dir_logits"],
    opset_version=17,
    dynamo=False,  # use legacy exporter for smaller files
)
print("Exported direction_model.onnx")

# Verify
import os
size_kb = os.path.getsize("web/direction_model.onnx") / 1024
print(f"  Size: {size_kb:.1f} KB")
