
# Load libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
import wandb


def main(name_model, model):

    wandb.init(project='iris', 
                group=name_model, # Group experiments by model
                reinit=True
    )

    # Load dataset
    df = datasets.load_iris()
    X = df.data 
    y = df.target

    # Split into train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)

    accuracy = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy').mean()
    f1_macro = cross_val_score(model, X_train, y_train, cv=kfold, scoring='f1_macro').mean()
    neg_log_loss = cross_val_score(model, X_train, y_train, cv=kfold, scoring='neg_log_loss').mean()

    wandb.log({'accuracy': accuracy,
                'f1_macro': f1_macro,
                'neg_log_loss': neg_log_loss})

if __name__=='__main__':
    models = {'LogisticRegression': LogisticRegression(solver='liblinear', multi_class='ovr'),
            'LinearDiscriminantAnalysis': LinearDiscriminantAnalysis(),
            'KNeighborsClassifier': KNeighborsClassifier(),
            'DecisionTreeClassifier': DecisionTreeClassifier(),
            'GaussianNB': GaussianNB()}
    
    for name, model in models.items():
        main(name, model)  




