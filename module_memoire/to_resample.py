def resampling_data(x,y,method="SMOTE"):
    
    seed=1234
    from imblearn.over_sampling import ADASYN, SMOTE
    if method=="SMOTE":
        resampler=SMOTE(sampling_strategy = 'auto' , random_state =seed , k_neighbors = 10)
        resampler.fit(x,y)
        x_resampled,y_resampled=resampler.fit_resample(x,y)
    else :
        resampler=ADASYN(sampling_strategy = 'auto' , random_state =seed , k_neighbors = 10)
        resampler.fit(x,y)
        x_resampled,y_resampled=resampler.fit_resample(x,y)
    if method=="sampling":
        test_data=pd.concat([x,y])
        minority=test_data[test_data.stroke==1]
        majority=test_data[test_data.stroke==0]
        minority_upsampled = resample(minority,replace=True,n_samples=len(majority), random_state=seed)
        resampler = pd.concat([majority,minority_upsampled])
        x_resampled= resampler.drop("stroke",axis=1)
        y_resampled= resampler["stroke"]
    return x_resampled,y_resampled,resampler
                 