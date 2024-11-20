def evaluation_of_model(model,features,label,model_name,model_resume):
    
    """
    This function is to compute the metrics of model and to save it.
    model : the model
    features : features for prediction (pd.DataFrame)
    model_name : str (name of model)
    model_resume : model_resume=pd.DataFrame(data=["precision0","recall0","f1_0","precision1","recall1","f1_1","accuracy_","precision_model","recall_model","f1_model","model_params"],columns=["metrics"])
    commentaires :str (to describe)
    """
    from sklearn.metrics import classification_report
    
    pred=model.predict(features)
    dict_result=classification_report(label,pred,output_dict=True)
    precision0=dict_result['0']['precision']
    recall0=dict_result['0']['recall']
    f1_0=dict_result['0']['f1-score']
    precision1=dict_result['1']['precision']
    recall1=dict_result['1']['recall']
    f1_1=dict_result['1']['f1-score']
    accuracy_=dict_result['accuracy']
    macro_precision=dict_result['macro avg']['precision']
    macro_recall=dict_result['macro avg']['recall']
    macro_f1=dict_result['macro avg']['f1-score']
    weighted_precision=dict_result['weighted avg']['precision']
    weighted_recall=dict_result['weighted avg']['recall']
    weighted_f1=dict_result['weighted avg']['f1-score']
    
    model_resume[model_name]=[precision0,recall0,f1_0,precision1,recall1,f1_1,accuracy_,macro_precision,macro_recall,macro_f1,weighted_precision,weighted_recall,weighted_f1,model]
    print(classification_report(label,pred))
    model_resume.to_csv(r"C:\Users\Ninette HOUKPONOU\Repertoire_python\Memoire\new_data\model_resume.csv")
    #print(1)
    return model_resume