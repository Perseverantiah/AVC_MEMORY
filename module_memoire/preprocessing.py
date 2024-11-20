# module preprocessing
def to_encoded(data,var_to_encode):
    """
    This function is to encode the categorical variable
    data : pd.DataFrame
    var_to_encoded: str (name of variable to encode)
    """
    data[var_to_encode] = data[var_to_encode].astype('category').cat.codes
    return data

def separation_of_train_test(data,label_name,size_=0.3):
    """
    This function is to separate a test_set and the train_set.
    data :pd.DataFrame
    label_name : str
    size_ : pourcentage of the test set
    
    return : x_tain,x_test and y_train, y_test
    """
    from sklearn.model_selection import train_test_split
    seed=1234
    y=data[label_name]
    x=data.drop(label_name,axis=1)
    X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=size_,random_state=seed,stratify=y)
    return X_train,X_test,y_train,y_test


def to_standardized(train_data):
    """
    This function allow you to standardized your data.
    return : data_standardized and the scaler to scale your test data
    """
    from sklearn.preprocessing import MinMaxScaler
    import pandas as pd
    scaler=MinMaxScaler()
    scaler.fit(train_data)
    return pd.DataFrame(scaler.transform(train_data),columns=train_data.columns),scaler


def features_selection(x_train,x_test,y_train,seuil=0.03,seed=1234):
    """
    This function purpose is to select the most important var
    x_train : pd.DataFrame
    x_test : pd.DataFrame
    seuil : threshold of importance of var
    return x_train,x_test
    """
    from sklearn.ensemble import RandomForestClassifier
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    rdf=RandomForestClassifier(random_state=seed)
    rdf.fit(x_train,y_train)
    plt.figure()
    features_important=pd.Series(rdf.feature_importances_, index=x_train.columns).sort_values(ascending=False)
    sns.barplot(x=features_important.index, y=features_important)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Variables")
    plt.ylabel("score of importance")
    plt.title("Importance of feature")
    plt.show()
    features_selected=features_important[features_important>seuil].index.to_list()
    x_train=x_train[features_selected]
    x_test=x_test[features_selected]
    return x_train,x_test