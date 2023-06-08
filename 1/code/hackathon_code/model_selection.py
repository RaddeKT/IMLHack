import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, accuracy_score
from preprocess import get_preprocessed_data

train_df = pd.read_csv("../datasets/agoda_cancellation_train.csv")

def decision_tree_selection(X_train, y_train, X_test, y_test, depth_values):
    scores = []

    for depth in depth_values:
        model = DecisionTreeClassifier(max_depth=depth)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        scores.append(f1_score(y_test, y_pred))

    plt.plot(depth_values, scores, marker='o')
    plt.xlabel('Maximum Depth')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs Maximum Depth')
    plt.show()


def random_forest_selection(X_train, y_train, X_test, y_test, depth_values):
    scores = []

    for depth in depth_values:
        model = RandomForestClassifier(max_depth=depth)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        scores.append(f1_score(y_test, y_pred))

    plt.plot(depth_values, scores, marker='o')
    plt.xlabel('Maximum Depth')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs Maximum Depth')
    plt.show()


df = get_preprocessed_data(train_df)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
X_train, y_train = train_df.drop("did_cancel", axis=1), train_df["did_cancel"]
X_test, y_test = test_df.drop("did_cancel", axis=1), test_df["did_cancel"]
random_forest_selection(X_train, y_train, X_test, y_test, [i for i in range(1, 20)])
