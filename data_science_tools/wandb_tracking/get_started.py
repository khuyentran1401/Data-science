import wandb
from sklearn.model_selection import cross_val_score
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


def main():
    # Initialize the project
    wandb.init(project='iris')

    # Load dataset
    df = datasets.load_iris()
    X = df.data 
    y = df.target

    # Split into train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

    # Initialize the model
    model = LogisticRegression(solver='liblinear', multi_class='ovr')

    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)

    # Get metrics
    accuracy = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy').mean()
    f1_macro = cross_val_score(model, X_train, y_train, cv=kfold, scoring='f1_macro').mean()
    neg_log_loss = cross_val_score(model, X_train, y_train, cv=kfold, scoring='neg_log_loss').mean()

    # Log the results
    wandb.log({'accuracy': accuracy,
                'f1_macro': f1_macro,
                'neg_log_loss': neg_log_loss})

if __name__ == '__main__':
    main()