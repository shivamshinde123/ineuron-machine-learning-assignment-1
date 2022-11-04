from sklearn.linear_model import LinearRegression
import pickle
import os
import pandas as pd

class ModelCreation:

    def __init__(self) -> None:
        pass

    def separate_data_into_dependent_and_independent_features(self,df):

        X = df.drop(columns=['price'])
        y = df['price']

        return X,y

    def create_and_save_model(self,df):

        X,y = self.separate_data_into_dependent_and_independent_features(df)

        lr = LinearRegression()
        model = lr.fit(X,y)

        if not os.path.exists('models'):
            os.makedirs('models')

        with open(os.path.join('models','lr_model.pkl'), 'wb') as f:
            pickle.dump(model,f)


if __name__ == '__main__':

    model_creation_obj = ModelCreation()

    df = pd.read_csv('processed_data/processed.csv')

    model_creation_obj.create_and_save_model(df)

 