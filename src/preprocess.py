from get_data import GetData
import numpy as np
import os


class Preprocess:

    def __init__(self) -> None:
        pass

    def getting_numerical_feature_names(self,df):
        lst = ['price','CHAS','RAD']
        num_feat = []
        for feature in df.columns:
            if feature not in lst:
                num_feat.append(feature)
        return num_feat

    def remove_unnecessary_feature(self,df,feature_name):

        """
        Removes the unnecessary features from the dataset.
        
            Parameters:
            df (dataframe object): The dataframe from which the unnecessary feature needs to be removed.
            feature_name (string): The name of feature to be removed from the dataset

            Returns: None
        """

        df = df.drop(columns=[feature_name],axis=1)
        return df

    
    def getting_quantile_info_about_data(self,df):

        num_feat = self.getting_numerical_feature_names(df)
        Q1_list = []
        Q3_list = []
        IQR_list = []

        for feature in num_feat:
            Q1 = np.round(np.percentile(df[feature],25),2)
            Q1_list.append(Q1)
            Q3 = np.round(np.percentile(df[feature],75),2)
            Q3_list.append(Q3)
            IQR = np.round(Q3 - Q1,2)
            IQR_list.append(IQR)

        return Q1_list, Q3_list, IQR_list

    
    def replace_outliers_with_null_values(self,df):

        num_feat = self.getting_numerical_feature_names(df)
        Q1_list, Q3_list, IQR_list = self.getting_quantile_info_about_data(df)

        for i in range(len(num_feat)):
            upper_limit = Q3_list[i] + 1.5*IQR_list[i]
            lower_limit = Q1_list[i] - 1.5*IQR_list[i]

            df[num_feat[i]] = np.where((df[num_feat[i]] > upper_limit) | (df[num_feat[i]] < lower_limit), np.nan, df[num_feat[i]])

            return df


    def replace_null_values_with_mean_of_the_feature(self,df):

        num_feat = self.getting_numerical_feature_names(df)

        for feature in ['CHAS','RAD']:
            df[feature].fillna(df[feature].mode(),inplace=True)

        for feature in num_feat:
            df[feature].fillna(df[feature].mean(),inplace=True)

        if not os.path.exists('processed_data'):
            os.makedirs('processed_data')

        df.to_csv('processed_data/processed.csv',index=False,columns=df.columns)


if __name__ == "__main__":
    
    preprocess_obj = Preprocess()

    df = GetData().get_data()

    df = preprocess_obj.remove_unnecessary_feature(df,"TAX")

    df = preprocess_obj.replace_outliers_with_null_values(df)

    preprocess_obj.replace_null_values_with_mean_of_the_feature(df)



