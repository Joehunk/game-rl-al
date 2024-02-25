import torch
import sys

# Check CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if f"{device}" != "cuda":
    print("WARNING: using CPU device which will be super slow!", file=sys.stderr)
