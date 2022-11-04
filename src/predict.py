import streamlit as st
import pickle
import pandas as pd
import numpy as np

class Prediction:


    def __init__(self) -> None:
        pass

    def Predict(self):

        with open('models/lr_model.pkl','rb') as f:
            model = pickle.load(f)

        st.title('Boston Price Prediction')

        st.write('Provide the values for following features')

        df = pd.read_csv('processed_data/processed.csv')
        print(df.columns)
        df_columns = np.delete(df.columns.to_numpy(),-1)
        print(df_columns)

        feature_min = []
        feature_max = []
        feature_mean = []
        input_from_user = []

        for feature in df_columns:
            feature_mean.append(df.describe().T[['mean','min','max']].loc[feature]['mean'])
            feature_min.append(df.describe().T[['mean','min','max']].loc[feature]['min'])
            feature_max.append(df.describe().T[['mean','min','max']].loc[feature]['max'])

        for index, feature in enumerate(df_columns):
            user_input = st.slider(feature,float(feature_min[index]),float(feature_max[index]),float(feature_mean[index]),0.1)
            input_from_user.append(user_input)

        input = np.array(input_from_user).reshape(1,-1)

        predict_btn = st.button("Predict Price")

        if predict_btn:
            lr_pred = model.predict(input)

            st.header('Predictions')

            st.write(f"Price: ${np.round(lr_pred[0],2)}")

            st.markdown("**Note that the prediction may be way of the actual price since the prediction made using linear regression which is one of the simplest machine learning algorithms.**")


if __name__ == '__main__':

    p = Prediction()
    p.Predict()

