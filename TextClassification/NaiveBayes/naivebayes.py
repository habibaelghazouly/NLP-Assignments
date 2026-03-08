import numpy as np
from collections import defaultdict

class NaiveBayes:
    
    def __init__(self):
        self.class_priors = {}
        self.word_counts = {}
        self.class_word_totals = {}
        self.vocab = set()
        self.classes = None

    def fit(self, X, y):
        """
        X : list of token lists
        y : labels
        """
        self.classes = np.unique(y)

        doc_count = len(y)

        # Initialize structures
        for c in self.classes:
            self.word_counts[c] = defaultdict(int)
            self.class_word_totals[c] = 0

        class_counts = defaultdict(int)

        # Count words
        for tokens, label in zip(X, y):
            class_counts[label] += 1

            for word in tokens:
                self.word_counts[label][word] += 1
                self.class_word_totals[label] += 1
                self.vocab.add(word)

        # Compute priors
        for c in self.classes:
            self.class_priors[c] = class_counts[c] / doc_count

        self.vocab_size = len(self.vocab)

    def predict(self, X):
        predictions = []

        for tokens in X:
            class_scores = {}

            for c in self.classes:

                # Start with log prior
                log_prob = np.log(self.class_priors[c])

                for word in tokens:

                    word_count = self.word_counts[c].get(word, 0)

                    # Laplace smoothing
                    prob = (word_count + 1) / (
                        self.class_word_totals[c] + self.vocab_size
                    )

                    log_prob += np.log(prob)

                class_scores[c] = log_prob

            predicted_class = max(class_scores, key=class_scores.get)
            predictions.append(predicted_class)

        return np.array(predictions)