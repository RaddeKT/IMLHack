import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression


def execute_task_1(train_df, test_df):
    ids = test_df["h_booking_id"]

    clean_train = preprocess(train_df)
    clean_test = preprocess(test_df, is_test=True)

    X_train, y_train = clean_train.drop("did_cancel", axis=1), clean_train["did_cancel"]

    model = LinearDiscriminantAnalysis()
    model.fit(X_train, y_train)
    y_pred = model.predict(clean_test)

    result_df = pd.DataFrame({"h_booking_id": ids, "did_cancel": y_pred})
    result_df.to_csv("../predictions/task1.csv", index=False)


def execute_task_2(train_df, test_df):
    ids = test_df["h_booking_id"]

    clean_train = preprocess(train_df)
    clean_test = preprocess(test_df, is_test=True)

    X_train, y_train = clean_train.drop("original_selling_amount", axis=1), clean_train["original_selling_amount"]

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(clean_test)


    result_df = pd.DataFrame({"h_booking_id": ids, "predicted_selling_amount": y_pred})
    result_df.to_csv("../predictions/task2.csv", index=False)


def main():
    task1_train = pd.read_csv("../datasets/agoda_cancellation_train.csv")
    task1_test = pd.read_csv("../datasets/Agoda_Test_1.csv")
    execute_task_1(task1_train, task1_test)

    task2_train = pd.read_csv("../datasets/agoda_cancellation_train.csv")
    task2_test = pd.read_csv("../datasets/Agoda_Test_2.csv")
    execute_task_2(task2_train, task2_test)


if __name__ == '__main__':
    main()
