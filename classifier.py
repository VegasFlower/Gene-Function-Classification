import pandas as pd
import csv
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score


def outlier_detection(df):
    for i in range(0, 103):
        max_value = df[df.columns[i]].mean() + 10 * df[df.columns[i]].std()
        min_value = df[df.columns[i]].mean() - 10 * df[df.columns[i]].std()
        new_df = df[(df[df.columns[i]] < max_value) & (df[df.columns[i]] > min_value) | (df[df.columns[i]].isna())]
        df = new_df
    return df


def imputation_knn(df):
    imputer = KNNImputer(n_neighbors=2, weights="uniform")
    for i in range(0, 103):
        df[df.columns[i]] = imputer.fit_transform(df[df.columns[i]].values.reshape(-1, 1))
    return df


def imputation_category(df):
    imputer = SimpleImputer(strategy="most_frequent")
    for i in range(103, 116):
        df[df.columns[i]] = imputer.fit_transform(df[df.columns[i]].values.reshape(-1, 1))
    return df


def normalize(df):
    normalizer = MinMaxScaler()
    for i in range(0, 103):
        df[df.columns[i]] = normalizer.fit_transform(df[df.columns[i]].values.reshape(-1, 1))
    return df


def pre_processing(df):
    df = outlier_detection(df)
    df = imputation_knn(df)
    df = imputation_category(df)
    df = normalize(df)
    return df


def main():
    # cross validations and evaluations on train_data
    df = pd.read_csv("Ecoli.csv")
    df = pre_processing(df)

    X = df.values[:, 0:116]
    y = df.values[:, 116:].round(decimals=0).astype(int)

    clf = DecisionTreeClassifier(criterion="gini", splitter="best", max_depth=5, min_samples_leaf=2)
    accuracy_scores = cross_val_score(clf, X, y, cv=10, scoring='accuracy')
    # print(accuracy_scores.mean())
    f1_scores = cross_val_score(clf, X, y, cv=10, scoring='f1')
    # print(f1_scores.mean())
    evaluation = ["%0.3f" % accuracy_scores.mean(), "%0.3f" % f1_scores.mean()]

    # predictions of test_data
    test_data = pd.read_csv("Ecoli_test.csv")
    test_data = pre_processing(test_data)

    clf = clf.fit(X, y)
    y_pred = clf.predict(test_data.values[:, :])

    # write predictions to the report
    df = pd.DataFrame(y_pred)
    df.to_csv('s4719268.csv', index=False, header=False)

    # write evaluations to the report
    f = open('s4719268.csv', 'a', newline='')
    writer = csv.writer(f)
    writer.writerow(evaluation)
    f.close()

if __name__ == "__main__":
    main()
