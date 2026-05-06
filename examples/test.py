import faulthandler
faulthandler.enable()

from pathlib import Path

import pandas as pd
import lsmash as ls

seq_path = Path("~/ZED/Research/zutil_/testsuite/seq.dat").expanduser()

# Read space-separated symbolic sequences
df = pd.read_csv(seq_path, sep=r"\s+", header=None, engine="python")

# Drop completely empty columns if present
df = df.dropna(axis=1, how="all")

# Convert to integer nested lists for lsmash
seqs = df.astype(int).values.tolist()

opt = ls.LsmashOptions()
opt.data_type = "symbolic"
opt.data_dir = "row"
opt.sae = True
opt.num_repeat = 20
opt.depth = 8

D = ls.from_sequences(seqs, opt)

print("Input shape:", df.shape)
print("Distance matrix shape:", D.shape)
print(D)
