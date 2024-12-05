import pandas as pd

splits = {'train': 'data/train-00000-of-00001-c6410a8bb202ca06.parquet', 'validation': 'data/validation-00000-of-00001-d21ad392180d1f79.parquet', 'test': 'data/test-00000-of-00001-d20b0e7149fa6eeb.parquet'}
df = pd.read_parquet("hf://datasets/bstee615/bigvul/" + splits["train"])

