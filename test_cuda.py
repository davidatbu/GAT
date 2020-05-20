import torch

print("device count", torch.cuda.device_count())
x = torch.randn(10, 10, device="cuda")
print(x)
