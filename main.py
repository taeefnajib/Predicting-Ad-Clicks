import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
from dataclasses_json import dataclass_json
from dataclasses import dataclass

### inspired from this tutorial: https://www.kaggle.com/code/pratikpal1/advertisement-click-predictions-using-logistic-reg

@dataclass_json
@dataclass
class Hyperparameters(object):
    filepath: str = "advertising.csv"
    test_size: float = 0.3
    random_state: int = 42

hp = Hyperparameters()

# Collecting data
def create_dataframe(filepath):
    return pd.read_csv(filepath)


# Splitting dataset into train and test datasets
def split_train_test(df, test_size, random_state):
    X = df.drop(['Clicked on Ad','Timestamp','Ad Topic Line','City','Country'],axis =1)
    y = df['Clicked on Ad']
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# Training model
def train_model(X_train, y_train):
    model = LogisticRegression()
    return model.fit(X_train,y_train)

# Creating prediction
def create_prediction(model, X_test, y_test):
    predictions = model.predict(X_test)
    print(accuracy_score(predictions, y_test))
    print(classification_report(y_test,predictions))
    print(confusion_matrix(y_test,predictions))
    return model

# Running workflow
def run_wf(hp: Hyperparameters) -> LogisticRegression:
    df = create_dataframe(filepath=hp.filepath)
    X_train, X_test, y_train, y_test = split_train_test(df=df, test_size=hp.test_size, random_state=hp.random_state)
    return train_model(X_train=X_train, y_train=y_train)
    # create_prediction(model=model, X_test=X_test, y_test=y_test)
    # return model

if __name__=="__main__":
    run_wf(hp=hp)