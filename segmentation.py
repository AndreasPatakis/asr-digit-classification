from sklearn.model_selection import train_test_split
from abc import ABC, abstractmethod
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import pickle

from utils import load_dataset, create_noisy_dataset


class BackgroundForegroundClassifier(ABC):
    '''An abstraction for creating Background vs Foreground classifiers.'''

    @abstractmethod
    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        '''
        fit(features=X_train, labels=y_train)

        Trains the classifier.

        Parameters
        ----------
        features : np.ndarray
            2D array containing the MFCC, Delta and Delta-Delta features
            of each audio.

        labels: np.ndarray
            Vector containing the corresponding labels (background/foreground).
        '''
        pass

    @abstractmethod
    def predict(self, features: np.ndarray) -> np.ndarray:
        '''
        predict(features=X)

        Classifies the given signals as background or foreground
        from the given features.

        Parameters
        ----------
        features : np.ndarray
            2D array containing the MFCC, Delta and Delta-Delta features
            of each audio.

        Returns
        -------
        labels : np.ndarray
            The predicted labels (background/foreground).
        '''
        pass

    @abstractmethod
    def score(self, test_features, test_labels) -> float:
        '''
        score(test_features=X_test, test_labels=y_test)

        Calculates the accuracy score of the classifier.

        Parameters
        ----------
        test_features : np.ndarray
            2D array containing the MFCC, Delta and Delta-Delta features
            of each test audio.

        test_labels: np.ndarray
            Vector containing the corresponding
            test labels (background/foreground).

        Returns
        -------
        score : float
            The accuracy score of the classifier.
        '''
        pass

    @abstractmethod
    def save(self):
        '''
        save()

        Saves the classifier as a pickle file.
        '''
        pass


class SVMBackgroundForegroundClassifier(BackgroundForegroundClassifier):
    '''
    A background vs foreground classifier
    that uses an SVM for sentence segmentation.
    '''

    def __init__(self) -> None:
        self.model = SVC()

    def fit(self, features, labels):
        self.model.fit(features, labels)

    def predict(self, features):
        return self.model.predict(features)

    def score(self, test_features, test_labels):
        return self.model.score(test_features, test_labels)

    def save(self):
        with open('svm_bf_model.pickle', 'wb') as pickle_out:
            pickle.dump(self, pickle_out)


if __name__ == '__main__':
    foreground_data = create_noisy_dataset(
        'data/background',
        'data/digits',
        1000
    )

    background_data = load_dataset('data/background', background=True)

    data = pd.concat((background_data, foreground_data), ignore_index=True)

    X_train, X_test, y_train, y_test = train_test_split(
        np.array(data.features.values.tolist()),
        np.array(data.label),
        test_size=0.2
    )

    classifier = SVMBackgroundForegroundClassifier()

    # Train the classifier
    classifier.fit(X_train, y_train)

    # Calculate the classifier's score
    score = classifier.score(X_test, y_test)
    print(f'Score: {score}')

    # Save the classifier
    classifier.save()
