import torch
import glob

model_path = '../models/*.pth'

for mp in glob.glob(model_path):
    print(mp)
    m = torch.load(mp)
    print(m.keys())
    if "ema" in m.keys():
        for k, v in m["ema"].items():
            m["model"][k.replace("model.", "")] = v
        del m["ema"]
        out_path = mp.replace(".pth", "_ema2model.pth")
        torch.save(m, out_path)
        print(out_path)