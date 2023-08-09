from sklearn.naive_bayes import GaussianNB
from raise_utils.learners.learner import Learner

class NaiveBayes(Learner):
    """Naive Bayes classifier"""
    def __init__(self, *args, **kwargs):
        """Initializes the classifier."""
        super(NaiveBayes, self).__init__(*args, **kwargs)
        self.learner = GaussianNB()
        self.random_map = {
            # This is where we set our random attributes
            "var_smoothing":[1e-10, 1e-09, 1e-08]
        }
        self._instantiate_random_vals()