import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import seaborn as sns
import matplotlib.pyplot as plt


class GestTrain:
    def __init__(self, data_path="gesture_data.csv"):
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.rf = None
        self.cvrf = None

    def load_data(self):
        self.df = pd.read_csv(self.data_path, header=None)
        
        # col 0 is label rest is 63 norm coords
        X = self.df.iloc[:, 1:]
        y = self.df.iloc[:, 0]
        
        print(f"total rows: {len(self.df)}")
        print(f"unique rows: {self.df.drop_duplicates().shape[0]}")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def train_models(self):
        # grid search with cross validation of 5 
        params = {
            "n_estimators":[100, 200, 300],
            "max_depth":[None, 10, 20],
            "min_samples_split": [2, 5],
        }
        grid = GridSearchCV(RandomForestClassifier(class_weight="balanced"), params, cv=5)
        grid.fit(self.X_train, self.y_train)
        print("best params:", grid.best_params_)
        self.cvrf = grid.best_estimator_

        # basic rf 
        self.rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
        self.rf.fit(self.X_train, self.y_train)

    def eval_models(self):
        preds = self.rf.predict(self.X_test)
        cvpred = self.cvrf.predict(self.X_test)

        print("base model")
        print(classification_report(self.y_test, preds))
        print("tuned model")
        print(classification_report(self.y_test, cvpred))

        self.save_cm(preds, self.rf.classes_, 'confusion_matrix.png', 'gesture recognition confusion matrix')
        
        # tuned one should have less errors on off diagonal
        self.save_cm(cvpred, self.cvrf.classes_, 'confusion_matrix_tuned.png', 'gesture recognition confusion matrix (tuned)')

    def save_cm(self, preds, classes, filename, title):
        cm = confusion_matrix(self.y_test, preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.xlabel('Predicted'); plt.ylabel('True')
        plt.title(title)
        plt.savefig(filename)
        print(f"saved {filename}")
        plt.close()

    def save_models(self):
        # save both runtime uses gesture_model_cv.pkl
        joblib.dump(self.rf, "gesture_model.pkl")
        joblib.dump(self.cvrf, "gesture_model_cv.pkl")
        print("models saved")

    def run(self):
        self.load_data()
        self.train_models()
        self.eval_models()
        self.save_models()


if __name__ == "__main__":
    gt = GestTrain()
    gt.run()
