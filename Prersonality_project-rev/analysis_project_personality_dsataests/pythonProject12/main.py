import  pandas as pd
from numpy.distutils.system_info import dfftw_info

file_path = "C:\Users\Muttaki\Downloads\archive\personality_dataset.csv"
df = pd.read_csv("C:\Users\Muttaki\Downloads\archive\personality_dataset.csv")
df_info = df.info()
df_head = df.head()
df_info,df_head
