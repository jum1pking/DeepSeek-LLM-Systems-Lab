import torch
import platform

print("Python:", platform.python_version())
print("PyTorch:", torch.__version__)
print("PyTorch CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("Compute capability:", torch.cuda.get_device_capability(0))

    x = torch.randn(
        (2048, 2048),
        device="cuda",
        dtype=torch.float16,
    )
    y = x @ x

    torch.cuda.synchronize()

    print("Matrix multiply:", y.shape)
    print("dtype:", y.dtype)
    print(
        "Allocated VRAM:",
        round(torch.cuda.memory_allocated() / 1024**3, 3),
        "GB",
    )

print("GPU TEST PASSED")
