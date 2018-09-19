from pathlib import Path
import pandas as pd
import sklearn.datasets

import os

data_dir = Path(os.path.dirname(os.path.abspath(__file__)))


def load_iris() -> pd.DataFrame:
    data_dict = sklearn.datasets.load_iris()
    data_dict.keys()
    df = pd.DataFrame(columns=data_dict['feature_names'],
                      data=data_dict['data'])
    df.columns = ['_'.join(tokens[:2]) for tokens in df.columns.str.split()]
    species_lookup = {n: v for n, v in enumerate(data_dict['target_names'])}
    df['species_code'] = data_dict['target']
    df['species'] = df.species_code.map(species_lookup)
    return df


if __name__ == '__main__':
    iris_data = load_iris()
    iris_data.to_csv(data_dir / 'iris.csv', index=False)
