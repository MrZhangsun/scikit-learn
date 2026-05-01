from pathlib import Path
import pandas as pd
data_path = Path(__file__).parent.parent.parent / 'resources/assets/boston_housing.csv'
data = pd.read_csv(data_path)
print(type(data))
print(data.loc[0])