import torch
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("torch", torch.__version__, "| device:", torch.cuda.get_device_name(0) if dev == "cuda" else "CPU ONLY")
gen = torch.Generator().manual_seed(1234)
x64 = torch.randn(8388608, generator=gen, dtype=torch.float64)
for name, dt in (("fp16", torch.float16), ("bf16", torch.bfloat16)):
    x = x64.to(dt)
    ref = x.double().cumsum(0)
    for n in (16, 1024, 8388608):
        c = torch.cumsum(x[:n], 0)
        g = torch.cumsum(x[:n].to(dev), 0).cpu()
        w = torch.cumsum(x[:n].to(dev), 0, dtype=torch.float32).to(dt).cpu()
        diff = (c != g)
        first = int(diff.nonzero()[0]) if diff.any() else -1
        print(f"{name} n={n:>8}: cpu!=dev at {int(diff.sum()):>8} elements, first at index {first}; "
              f"cpu!=dev(dtype=fp32) at {int((c != w).sum())}")
    c = torch.cumsum(x, 0); g = torch.cumsum(x.to(dev), 0).cpu()
    print(f"{name} element 15: true {ref[15].item():.8f} | cpu {c[15].item():.8f} | dev {g[15].item():.8f}")
    print(f"{name} first 16 inputs: {[round(v, 6) for v in x[:16].double().tolist()]}")
