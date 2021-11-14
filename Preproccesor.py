import seaborn as sns
import pickle
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from TrainModelForSendPoliceCars import *

# the path of  the original data
DATA_PATH = "Dataset_crimes.csv"
# fraction to use as train
TRAIN_AMOUNT = 0.7
YEAR = "Year"
LOCATION = "Location"
# Rsponce maps
INVERTED_RESPONSE_MAP = {0: "BATTERY", 1: 'THEFT', 2: "CRIMINAL DAMAGE", 3: 'DECEPTIVE PRACTICE', 4: "ASSAULT"}
RESPONSE_MAP = {"BATTERY": 0, 'THEFT': 1, "CRIMINAL DAMAGE": 2, 'DECEPTIVE PRACTICE': 3, "ASSAULT": 4}
# features that are not used at the learning process:
CANT_USE = ["FBI Code", "IUCR", "Description"]
DROP_LIST = ["ID", LOCATION, "Longitude", "Updated On", "Latitude", "X Coordinate", "Y Coordinate", "Community Area",
             "Beat", "Ward", "Year", "Case Number", "Block", "Unnamed: 0"]
# extracted new fetures:
LIST_HOT_WORDS = ["APARTMENT", "RESIDENCE", "STREET"]


def preprocess(data):
    """
    preprocess the data
    """
    # drop nun values
    data.dropna(inplace=True)
    # drop unnecessary features:
    data.drop(columns=DROP_LIST, inplace=True)
    data.drop(columns=CANT_USE, inplace=True)
    data.drop_duplicates(inplace=True)
    data.replace(to_replace=RESPONSE_MAP, inplace=True)
    # parse the hour
    date = pd.to_datetime(data['Date'])
    data["HOUR"] = date.dt.hour
    data.drop(columns=["Date"], inplace=True)
    # crime locations that have high correlation to the crime

    for key_word in LIST_HOT_WORDS:
        data[key_word] = data["Location Description"] == key_word
        #              * data[data["Location Description"] == key_word][
        # "Primary Type"].mean()
    for dummy_word in ["SIDEWALK", "SMALL RETAIL STORE", "DEPARTMENT STORE"]:
        data[dummy_word] = data["Location Description"] == dummy_word
    data.drop(columns=["Location Description"], inplace=True)
    # create dummies for the Districts:
    for dummy in range(26):
        data[f"District_{dummy}"] = data["District"] == dummy
    # convertse string that represents boolean value to 1 or 2
    data.replace(to_replace={"False": 0, "True": 1}, inplace=True)
    return data


def plot_amount(train):
    """plots location descirptions by amount"""
    plt.figure(figsize=(14, 10))
    plt.title('Amount of Crimes by  location')
    plt.ylabel('Crime Type')
    train.groupby([train["Location Description"]]).size().sort_values(ascending=True).plot(kind='barh')
    plt.show()


class ModelProcessor:
    """
    class that preprocess the code and used for further statics of the data
    """

    def __init__(self, data_path=DATA_PATH, to_train=False):
        if to_train:
            df1 = pd.read_csv("crimes_dataset_part2.csv")
            df2 = pd.read_csv("Dataset_crimes.csv")
            self.total_df = pd.concat([df1, df2], axis=0, ignore_index=True)
            self.total_df = preprocess(self.total_df)
            self.train = self.total_df.sample(frac=TRAIN_AMOUNT, random_state=200)  # random state is a seed value
            temp = self.total_df.drop(self.train.index)
            self.validation = temp.sample(frac=0.5, random_state=200)  # random state is a seed value
            self.test = temp.drop(self.validation.index)
        else:
            self.total_df = preprocess(pd.read_csv(data_path))

    def stats(self):
        """
        prints the descirption of the train data
        """
        for col in self.train.columns:
            print(col)
            print(self.train[col].describe())
            print("___________________________________________________________________________")
            print()
            print("___________________________________________________________________________")

    def split_to_X_y(self, data, index):
        """
        splits Chicago crime DataFrame to respones by Primary Type not including
        :param data:
        :param index:
        :return:
        """
        for i in range(5):
            if i != index:
                data = data.drop(columns=[f'Primary Type_{i}'])
        return data.drop(columns=[f'Primary Type_{index}']), data[f'Primary Type_{index}']

    def evaluation(self):
        """
        evaluates the best model, then serializes the best model.
        :return:
        """
        best_i, best_j, best_score = -float("inf"), -float("inf"), -float("inf")
        best_model = None
        for i in range(1, 20):
            for j in range(5, 30):
                model = DecisionTreeClassifier(max_depth=i, max_features=j)
                model.fit(self.train.drop(columns=['Primary Type']).values, self.train['Primary Type'].values)
                score = model.score(self.validation.drop(columns=['Primary Type']).values,
                                    self.validation['Primary Type'].values)
                if score > best_score:
                    test_score = model.score(self.test.drop(columns=['Primary Type']).values,
                                             self.test['Primary Type'].values)
                    best_score = score
                    best_i = i
                    best_j = j
                    best_model = model
        print(f"best so far {best_score} {best_i} {best_j} test:{test_score}")
        # Random tree evaluation
        for i in range(1, 20):
            for j in range(1, 10):
                model = RandomForestClassifier(max_depth=i, max_samples=1 / (j + 1))
                model.fit(self.train.drop(columns=['Primary Type']).values, self.train['Primary Type'].values)
                score = model.score(self.validation.drop(columns=['Primary Type']).values,
                                    self.validation['Primary Type'].values)
                print(f"Random best so far {best_score} {best_i} {best_j} test:{test_score}")
                if score > best_score:
                    test_score = model.score(self.test.drop(columns=['Primary Type']).values,
                                             self.test['Primary Type'].values)
                    best_score = score
                    best_i = i
                    best_j = 1 / (j + 1)
                    best_model = model
        print(f"Random best so far {best_score} {best_i} {best_j} test:{test_score}")
        return best_model


class MyTree:
    """
    class for multi class classification, this class uses the next algorithm:
    uses 5 classifiers to evaluate new data matrix and uses another classifier to train on the
    evaluated data matrix
    """

    def __init__(self, train, model):
        X = train.drop(columns=["Primary Type"])
        y = train["Primary Type"]
        train = pd.get_dummies(train, columns=["Primary Type"])
        data = np.ones(shape=(len(X), 10))
        self.trees = [model()]
        for i in range(10):
            self.trees.append(RandomForestClassifier(max_depth=12))
            temp_X, temp_Y = self.split_to_X_y(train, i % 5)
            self.trees[i].fit(temp_X, temp_Y)
            data[:, i] = self.trees[i].predict(X)
        self.mainTree = RandomForestClassifier(max_depth=12)
        self.mainTree.fit(data, y)

    def predict(self, X):
        data = np.ones(shape=(len(X), 10))
        for i in range(10):
            data[:, i] = self.trees[i].predict(X.values)
        return self.mainTree.predict(data)

    def score(self, data):
        """
        returns the accuracy score dot the given data
        :param data: data to calculate his accuracy
        """
        X = data.drop(columns=["Primary Type"])
        y = data["Primary Type"]
        return sum(self.predict(X) == y.values) / len(y)

    def split_to_X_y(self, data, index):
        for i in range(5):
            if i != index:
                data = data.drop(columns=[f'Primary Type_{i}'])
        return data.drop(columns=[f'Primary Type_{index}']).values, data[f'Primary Type_{index}'].values


class Serialization:
    def __init__(self, classifier, police_cars):
        self.classifier = classifier
        self.police_cars = police_cars
