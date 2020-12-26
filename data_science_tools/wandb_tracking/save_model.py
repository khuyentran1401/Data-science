
# Load libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import wandb
import pickle


def main(name_model, model):

    wandb.init(project='iris', 
                group=name_model, # Group experiments by model
    )

    # Load dataset
    df = datasets.load_iris()
    X = df.data 
    y = df.target
    features = df.feature_names

    # Split into train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

    # Train model, get predictions
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred = model.predict(X_test)
    y_probas = model.predict_proba(X_test)

    # Visualize all classification plots
    wandb.sklearn.plot_classifier(model, X_train, X_test, y_train, y_test, 
                                y_pred, y_probas, features, model_name=name_model)

    # Save model
    pickle.dump(model, open('model.pkl', 'wb'))

    wandb.save('model.pkl')


if __name__=='__main__':
   
    
    main('LinearDiscriminantAnalysis', LinearDiscriminantAnalysis())  




