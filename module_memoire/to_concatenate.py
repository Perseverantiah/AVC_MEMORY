def to_concatenate(df1, df2,list_of_var=["gender","hypertension","heart_disease","ever_married","work_type","Residence_type","smoking_status"]):
    """list_of_var : list (list of var of df1)
    df1 : pd.DataFrame. 
    df2 : pd.DataFrame
    
    return x_final : pd.DataFrame
    """
    import pandas as pd
    x_final=pd.concat([df1[list_of_var].reset_index(drop=True),df2],axis=1)
    return x_final