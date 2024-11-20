#module exploratory analysis
def import_data(path):
    """
    This function import the data. You just need to enter the path
    path : str
    return : DataFrame
    """
    import pandas as pd
    data=pd.read_csv(path)
    return data


def sanity_check(data):
    """
    This function check the sanity of the data. We check that we don't have a missing vallue or nul
    data : pd.DataFrame
    return : DataFrame
    """
    if (data.isna().sum).sum()>0 :
        data=data.dropna()
    return data



def basic_compute(data):
    print(data.head(10))
    print("----------------------------------------------------------------------------------------------")
    print("The shape of your dataset {}".format(data.shape))
    print("----------------------------------------------------------------------------------------------")
    data.info()
    print("----------------------------------------------------------------------------------------------")
    print(data.describe())
    print("----------------------------------------------------------------------------------------------")
    numerical_columns=[]
    for col in data.select_dtypes('float64'):
        numerical_columns.append(col)
    print ("numerical var {}".format(numerical_columns))
    print("----------------------------------------------------------------------------------------------")
    
    categorical_columns=[]
    for col in data.select_dtypes('object'):
        categorical_columns.append(col)
    for col in data.select_dtypes('int64'):
        categorical_columns.append(col)
    print ("categorical var {}".format(categorical_columns))
    