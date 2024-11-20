def compute_model(model_name,x_train,y_train):
    """
    This function allows you to compute different sort of model.
    You have RandomForest,XGBoost, BalancedBaggingClassifier or BalancedRandomForestClassifier.
    model_name : (strg) The possible value of model_name is "rdf" for RandomForest, "xgb" for XGBoost, "bbc" for BalancedBaggingClassifier and "brdf" for BalancedRandomForestClassifier
    
    x_train : pd.DataFrame
    y_train : Series
    
    return model
    """
    
    seed=1234
    #importation of modules
    from imblearn.ensemble import BalancedRandomForestClassifier
    from imblearn.ensemble import BalancedBaggingClassifier
    from imblearn.ensemble import RUSBoostClassifier
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import classification_report
    
    
    class_weight=y_train[y_train==0].size/y_train[y_train==1].size
    if model_name=="rdf":
        enter_param=str(input("Do you want to enter the params. Y/N"))
        if (enter_param=='Y'):
            
            # n_estimators
            n_estimators_list = input("enter the list of n_estimators that you want to test separated by espace : ")
            n_estimators_list = n_estimators_list.split()
            n_estimators = [int(num) for num in n_estimators_list]
            
            #min_samples
            
            
            min_samples_split_list = input("enter the list of min_samples_split that you want to test separated by espace : ")
            min_samples_split_list = min_samples_split_list.split()
            min_samples_split = [int(num) for num in min_samples_split_list]
            
            
            # max_leaf_nodes
            max_leaf_nodes_list = input("enter the list of max_leaf_nodes that you want to test separated by espace : ")
            max_leaf_nodes_list =max_leaf_nodes_list.split()
            max_leaf_nodes = [int(num) for num in max_leaf_nodes_list]
            
            
            # max_depth
            max_depth_list = input("enter the list of max_leaf_nodes that you want to test separated by espace : ")
            max_depth_list = max_depth_list.split()
            max_depth = [int(num) for num in max_depth_list]

            
           
            
            param_grid=param_grid={'n_estimators' : n_estimators,
            'min_samples_split' : min_samples_split,
            'max_leaf_nodes': max_leaf_nodes,
            #'criterion' : criterion,
            'max_depth' : max_depth}
            
            print(param_grid)
        else :
            param_grid={'n_estimators' : [500,300,466,876],
            'min_samples_split' :[10,47,38,29,48],
            'max_leaf_nodes':[2,39],
            #'criterion' : ["absolute_error"],
            'max_depth' : [20,34]}
            
        model=GridSearchCV(estimator=RandomForestClassifier(class_weight={0:1,1:class_weight},random_state=seed,n_jobs=-1), param_grid=param_grid,scoring='f1',cv=5)
        
    if model_name=="xgb":
        enter_param=str(input("Do you want to enter the params. Y/N"))
        if (enter_param=='Y'):
            n_estimators=list(input("enter the list of n_estimators that you want to test"))
            base_score=list(input("enter the list of base_score that you want to test"))
            eval_metrics=list(input("enter the list of eval_metrics that you want to test"))
            learning_rate=list(input("enter the list of learning_rate that you want to test"))
            max_depth=list(input("enter the list of max_depth that you want to test"))
            
            param_grid={
                'base_score': base_score,
                'n_estimators':n_estimators,
                'eval_metrics': eval_metrics,
                'booster' : booster,
                'max_depth' : max_depth,
                'learning_rate' : learning_rate
            }
            
        else :
            param_grid={
                'base_score': [0.5,0.1],
                'n_estimators':[500,1000],
                'eval_metrics': ['auc','aucpr'],
                'booster' : ['gbtree'],
                'max_depth' : [10,20],
                'learning_rate' : [0.01,0.001]
            }
            
        model = GridSearchCV(estimator=XGBClassifier(random_state=seed,scale_pos_weight=class_weight), param_grid=param_grid,scoring='f1',cv=5)
    
    if model_name=="bbc":
        enter_param=str(input("Do you want to enter the params. Y/N"))
        if (enter_param=='Y'):
            n_estimators=list(input("enter the list of n_estimators that you want to test"))
            
            param_grid={
                'n_estimators':n_estimators,
            }
            
        else :
            param_grid={
                'n_estimators':[200,500],
                #'max_depth' : [10,20],
                #'max_features' : [10,20],
                #'sample_strategy': ['not majority','all']
            }
        model=GridSearchCV(estimator=BalancedBaggingClassifier(random_state=seed),param_grid=param_grid,scoring='f1',cv=5)
        
    if model_name=="brdf":
        param_grid={
                'n_estimators':[200,500],
                'min_samples_split'  :[10],
                'max_leaf_nodes':[2],
                #'criterion' : ["log_loss"],
                'max_depth' : [20],
                #'sample_strategy ': ['not majority','all']
            }
        model=GridSearchCV(estimator=BalancedRandomForestClassifier(class_weight={0:1,1:class_weight},random_state=seed), param_grid=param_grid,scoring='f1',cv=5)
    model.fit(x_train,y_train)
            
    return model
