import argparse
import pandas as pd
from doduo import Doduo, AnnotatedDataFrame  # adjust import as needed
import math

# 1) load and row‑truncate
df1 = pd.read_csv(
    "sample_tables/cholera.csv",
    sep=';',
    skiprows=1,
    dtype=str
)
# TODO: Note that it may work incorrectly, it should be tested out.
max_rows = 3
df_small = df1.iloc[:max_rows]

# 2) set up your Doduo instance
args = argparse.Namespace(model="wikitable")
doduo = Doduo(args=args)

# 3) decide how many column‑chunks to use
num_chunks = 10
n_cols = df_small.shape[1]
chunk_size = math.ceil(n_cols / num_chunks)

# 4) annotate each chunk separately
adf_chunks = []
for start in range(0, n_cols, chunk_size):
    chunk = df_small.iloc[:, start:start+chunk_size]
    adf_chunks.append(doduo.annotate_columns(chunk))

# 5) re‑combine into one AnnotatedDataFrame
final_adf = AnnotatedDataFrame(df_small)

# flatten and concatenate all embeddings, types, (and relations, if any)
final_adf.colemb   = [emb for adf in adf_chunks for emb in adf.colemb]
final_adf.coltypes = [typ for adf in adf_chunks for typ in adf.coltypes]

if hasattr(adf_chunks[0], "colrels"):
    final_adf.colrels = [rel for adf in adf_chunks for rel in adf.colrels]

col_name_to_type = dict(zip(df_small.columns, final_adf.coltypes))

# 2) print them in the desired “original -> prediction” format
for col, typ in col_name_to_type.items():
    print(f"{col} -> {typ}")


# now you have full‑table predictions  …
#print(final_adf.coltypes)
#if hasattr(final_adf, "colrels"):
#    print(final_adf.colrels)