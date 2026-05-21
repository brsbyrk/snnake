"""Export sNNake v5 components to ONNX for web inference."""
import sys
sys.path.insert(0, "src")

import torch
import torch.nn as nn
from snnake.model_v5 import StructuredWorldModel, GRID_SIZE, NUM_CELLS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
model = StructuredWorldModel().to(device)
state = torch.load("checkpoints_v5/best.pt", map_location=device)
model.load_state_dict(state)
model.eval()
print(f"Model loaded ({model.get_num_params():,} params)")

# ─── 1. Direction MLP ───
# Wrapper that takes (direction, action) → dir_logits (4-class)
class DirNetWrapper(nn.Module):
    def __init__(self, dir_net):
        super().__init__()
        self.dir_net = dir_net
    def forward(self, direction, action):
        x = torch.cat([direction, action], dim=1)
        return self.dir_net(x)

dir_model = DirNetWrapper(model.dir_net).to(device)
dir_model.eval()

# Export
dummy_dir = torch.randn(1, 4, device=device)
dummy_act = torch.randn(1, 3, device=device)

torch.onnx.export(
    dir_model,
    (dummy_dir, dummy_act),
    "web/direction_model.onnx",
    input_names=["direction", "action"],
    output_names=["dir_logits"],
    dynamic_axes={
        "direction": {0: "batch"},
        "action": {0: "batch"},
        "dir_logits": {0: "batch"},
    },
    opset_version=14,
)
print("✓ Exported direction_model.onnx")

# ─── 2. Food Predictor ───
# Takes body positions + mask → food_logits (100-class)
class FoodNetWrapper(nn.Module):
    def __init__(self, body_encoder, food_net):
        super().__init__()
        self.body_encoder = body_encoder
        self.food_net = food_net
    def forward(self, body, body_mask):
        ctx = self.body_encoder(body, body_mask)
        return self.food_net(ctx)

food_model = FoodNetWrapper(model.body_encoder, model.food_net).to(device)
food_model.eval()

MAX_BODY_LEN = 40
dummy_body = torch.randn(1, MAX_BODY_LEN, 2, device=device)
dummy_mask = torch.ones(1, MAX_BODY_LEN, device=device)

torch.onnx.export(
    food_model,
    (dummy_body, dummy_mask),
    "web/food_model.onnx",
    input_names=["body", "body_mask"],
    output_names=["food_logits"],
    dynamic_axes={
        "body": {0: "batch"},
        "body_mask": {0: "batch"},
        "food_logits": {0: "batch"},
    },
    opset_version=14,
)
print("✓ Exported food_model.onnx")

# ─── 3. Combined (direction + food) ───
class CombinedModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.dir_net = model.dir_net
        self.body_encoder = model.body_encoder
        self.food_net = model.food_net
    def forward(self, direction, action, body, body_mask):
        # Direction
        x = torch.cat([direction, action], dim=1)
        dir_logits = self.dir_net(x)
        # Food
        ctx = self.body_encoder(body, body_mask)
        food_logits = self.food_net(ctx)
        return dir_logits, food_logits

combined = CombinedModel(model).to(device)
combined.eval()

torch.onnx.export(
    combined,
    (dummy_dir, dummy_act, dummy_body, dummy_mask),
    "web/combined_model.onnx",
    input_names=["direction", "action", "body", "body_mask"],
    output_names=["dir_logits", "food_logits"],
    dynamic_axes={
        "direction": {0: "batch"},
        "action": {0: "batch"},
        "body": {0: "batch"},
        "body_mask": {0: "batch"},
        "dir_logits": {0: "batch"},
        "food_logits": {0: "batch"},
    },
    opset_version=14,
)
print("✓ Exported combined_model.onnx")

# Verify file sizes
import os
for f in os.listdir("web/"):
    if f.endswith(".onnx"):
        size_kb = os.path.getsize(f"web/{f}") / 1024
        print(f"  {f}: {size_kb:.1f} KB")
