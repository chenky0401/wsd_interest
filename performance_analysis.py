"""
This module is for testing the averge performance of different classifiers and taggers.
To test the classifiers, modify the features used in training in wsd_features(instance).
To test the tagger, modify the training model in train_tagger(train_sents).

author: Kuan-Yu Chen
email: chenky@brandeis.edu
"""

import nltk
from interest import create_labeled_data, train_test_split, train_tagger, train_classifier

def wsd_features(instance):
    """Extract feature from the given instance of SensevalInstance and return a dictionary.
    """
    features = {}
    pos = instance.position
    words = {w for (w, t) in instance.context}

    w, t = instance.context[pos-1]
    features["previous_word"] = w
    features["previous_word_tag"] = t
    
    w, t = instance.context[pos+1]
    features["following_word"] = w 
    features["following_word_tag"] = t

    features["position"] = pos
    features["contains(%)"] = "%" in words
    features["contains($)"] = "$" in words
    features["is_plural"] = instance.context[pos][0][-1] == "s"
    
    return features

def train_tagger(train_sents):
    """Train and return a tagger using train_sents.
    """
    tags = [t for sent in train_sents for (w, t) in sent ]
    t0 = nltk.DefaultTagger(nltk.FreqDist(tags).max())
    t1 = nltk.UnigramTagger(train_sents, backoff=t0)
    t2 = nltk.BigramTagger(train_sents, backoff=t1)
    return t2


if __name__ == '__main__':
    # List of classifiers
    classifiers = (
        nltk.DecisionTreeClassifier,
        nltk.NaiveBayesClassifier,
        )

    # Collect data
    labeled_data = create_labeled_data()
    total_feature_sets = [(wsd_features(inst), label) for (inst, label) in labeled_data]
    tagged_sents = [inst.context for (inst, label) in labeled_data]


    num_trials = 100

    # Classifier testing
    for model in classifiers:
        print(model)
        train_score = []
        test_score = []
        for i in range(num_trials):
            if i%10 == 0:
                print(i)
            train_set, test_set = train_test_split(total_feature_sets, shuffle=True)
            classifier = model.train(train_set)
            train_score.append(nltk.classify.accuracy(classifier, train_set))
            test_score.append(nltk.classify.accuracy(classifier, test_set))
            # classifier.show_most_informative_features(10)
            # print(classifier.pseudocode(depth=2))
        print("Avg training score:", sum(train_score) / num_trials)
        print("Avg testing score:", sum(test_score) / num_trials)
        

    # Tagger testing
    tagger_score = []
    for i in range(num_trials):
        if i%10 == 0:
                print(i)
        train_sents, test_sents = train_test_split(tagged_sents, shuffle=True)
        tagger = train_tagger(train_sents)
        tagger_score.append(tagger.evaluate(test_sents))
    print("Avg tagger score:", sum(tagger_score) / num_trials)
