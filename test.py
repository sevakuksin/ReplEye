import torch
print(torch.cuda.is_available())  # Should return True if CUDA is installed and available
print(torch.cuda.device_count())  # Should return the number of GPUs available
