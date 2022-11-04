import pandas as pd
from sklearn.datasets import load_boston


class GetData:


    def __init__(self) -> None:
        pass

    def get_data(self):
        boston = load_boston()
        bos = pd.DataFrame(boston.data, columns=boston.feature_names)
        bos['price'] = boston.target
        return bos
