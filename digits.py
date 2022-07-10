from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from abc import ABC, abstractmethod
import numpy as np

from utils import load_dataset


class DigitClassifier(ABC):
    '''An abstraction for creating digit classifiers.'''

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
            Vector containing the corresponding labels (digit values).
        '''
        pass

    @abstractmethod
    def predict(self, features: np.ndarray) -> np.ndarray:
        '''
        predict(features=X)

        Predicts the digits from the given features.

        Parameters
        ----------
        features : np.ndarray
            2D array containing the MFCC, Delta and Delta-Delta features
            of each audio.

        Returns
        -------
        labels : np.ndarray
            The predicted labels (digits).
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
            Vector containing the corresponding test labels (digit values).

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


class KNNDigitClassifier(DigitClassifier):
    '''A digit classifier that uses K-nearest neighbors.'''

    def __init__(self, n_neighbors) -> None:
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)

    def fit(self, features, labels):
        self.model.fit(features, labels)

    def predict(self, features):
        return self.model.predict(features)

    def score(self, test_features, test_labels):
        return self.model.score(test_features, test_labels)

    def save(self):
        # TODO Save the classifier
        pass


if __name__ == '__main__':
    data = load_dataset('data/digits')
    data.dropna(inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(
        np.array(data.features.values.tolist()),
        np.array(data.label),
        test_size=0.2
    )

    classifier = KNNDigitClassifier(n_neighbors=3)

    # Train the classifier
    classifier.fit(X_train, y_train)

    # Calculate the classifier's score
    score = classifier.score(X_test, y_test)
    print(f'Score: {score}')

    # Save the classifier
    classifier.save()
