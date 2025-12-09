import torch

a = torch.load("raw_full_model.pt")
b = torch.load("raw_split.pt")

print("shape full :", a.shape)
print("shape split:", b.shape)
print("same shape?", a.shape == b.shape)

if a.shape == b.shape:
    diff = (a - b).abs()
    print("max abs diff :", diff.max().item())
    print("mean abs diff:", diff.mean().item())
