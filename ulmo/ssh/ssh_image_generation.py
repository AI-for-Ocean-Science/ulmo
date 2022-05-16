import pandas as pd
import seaborn as sns

ds = pd.read_parquet(r"C:\Users\btpri\OneDrive\Desktop\SSH_std.parquet")

#print(ds)

sns.histplot(ds, x="LL")







