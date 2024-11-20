def plot_dependencing(model,x_train,model_name,features):
    """
    model: estimator 
    x_train : pd.DataFrame
    
    model_name : str (the name of model)
    
    features : features to plot the dependency
    """
    from sklearn.inspection import PartialDependenceDisplay
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(12,6))
    ax.set_title(model_name)
    plot_model=PartialDependenceDisplay.from_estimator(model,x_train,features, ax=ax)
    plt.show()