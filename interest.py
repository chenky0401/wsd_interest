"""
This module is an interactive program that demonstrates the results of running a word sense classifier on sentences from the Gutenberg corpus. The classifier is trained with the 2368 sentences that contain the noun 'interest' in the NLTK SenseEval 2 Corpus and can determine what sense of the noun `interest` is expressed in a sentence.

You can use the parameters eval, show_d, show_n in the train_classifier function to see the training result
of the classifiers and the parameter eval in train_tagger to see the performance of the tagger.


author: Kuan-Yu Chen
email: chenky@brandeis.edu

"""

import random
import re
import nltk
from nltk.corpus import senseval
from nltk.corpus.reader.senseval import SensevalInstance


def create_labeled_data():
    """Collect data from SenseEval 2 Corpus and create labeled data.
    """
    interest = senseval.instances('interest.pos')
    labeled_data = [(inst, inst.senses[0][-2:]) for inst in interest]
    
    return labeled_data


def train_test_split(data, ratio=0.75, shuffle=False):
    """Split the given data into two lists. 
    The parameter ratio indicates len(first list) / len(data).
    If shuffle is True, shuffles the data.
    """
    if shuffle:
        random.shuffle(data)

    split_point = int(len(data) * ratio)
    train_set = data[:split_point]
    test_set = data[split_point:]
    
    return train_set, test_set 

    
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
    # features["following_word_tag"] = t

    features["position"] = pos
    features["contains(%)"] = "%" in words
    features["contains($)"] = "$" in words
    features["is_plural"] = instance.context[pos][0][-1] == "s"
    
    return features


def make_instance(tagged_sentence):
    """Take a tagged sentence (a list of tuples) and convert it into SensevalInstance.
    """
    words = [t[0] for t in tagged_sentence]
    position = words.index('interest')
    return SensevalInstance('interest-n', position, tagged_sentence, [])


def train_tagger(train_sents, test_sents=None, eval=False):
    """Train and return a tagger using train_sents.
    If test_sents is also provided and eval=True, prints out the performance of the tagger.

    The average performance score of UnigramTagger without any back-off is already 0.900 (100 trials).
    With UnigramTagger backing off to DefaultTagger, the average score goes up to 0.926 (100 trials).
    Adding BigramTagger makes the average score go up to 0.936 (100 trials).
    """
    tags = [t for sent in train_sents for (w, t) in sent ]
    t0 = nltk.DefaultTagger(nltk.FreqDist(tags).max())
    t1 = nltk.UnigramTagger(train_sents, backoff=t0)
    t2 = nltk.BigramTagger(train_sents, backoff=t1)
    print("Done.")

    if test_sents and eval:
        print("Tagger performance: %.2f\n" % t2.evaluate(test_sents))
    return t2


def train_classifier(classifier, train_set, test_set=None, eval=False, show_d=False, show_n=False):
    """Train the classifier with train_set. 
    If test_set is also provided and eval=True, prints out the accuracy of classifier.
    If show_d=True, shows the first level of the Decision Tree classifier.
    If show_n=True, shows the 10 most informative features of the Naive Bayes classifier.


    [Feature analysis of Decision Tree classifier]
    Average train set accuracy with all features: 0.994 (100 trials)
    Average test set accuracy with all features: 0.790 (100 trials)

    Printing out the first level of the decision tree shows that the feature at all the stumps is the previous_word feature.
    Excluding the previous_word feature would decrease the test set accuracy to 0.727 (100 trials) and make the previous_word_tag 
    feature be at the first level. Taking out all the features about previous/following word/tag would lead to is_plural on the top 
    and a score of 0.59 (100 trials).


    [Feature analysis of Naive Bayes classifier:]
    Average train set accuracy with all features: 0.916 (100 trials)
    Average test set accuracy with all features: 0.863 (100 trials)
    
    Printing out the 10 most informative features shows that the most informative feature is following_word_tag = 'NNS'.
    However, excluding the following_word_tag feature would increase the test set accuracy to 0.878 (100 trials) with 
    is_plural and following_word features being the most informative features alternitively.
    While the feature contains(%) and features about previous/following word are generally informative, 
    the contain($) and position feature are rarely informative.

    """
    trained_classifier = classifier.train(train_set)
    print("Done.")

    if test_set and eval:
        print("train_set accuracy: %.2f" % nltk.classify.accuracy(trained_classifier, train_set))
        print("test_set accuracy: %.2f\n" % nltk.classify.accuracy(trained_classifier, test_set))
    if show_d:
        try:
            print(trained_classifier.pseudocode(depth=1))
        except:
            pass
    if show_n:
        try:
            trained_classifier.show_most_informative_features(10)
            print()
        except:
            pass

    return trained_classifier
            

def run_classifier(classifier, sent):
    """Run the classifier on the SensevalInstance instance, sent, and store the label in sent.senses.
    Return the label.
    """
    sent.senses = (classifier.classify(wsd_features(sent)), )
    return sent.senses[0]


def compare_classifiers(c1, c2):
    """Compare two lists and return the indices where the elements are different.
    """
    diff = []
    for i in range(len(c1[1])):
        if not c1[1][i] == c2[1][i]:
            diff.append(i)
    return diff



if __name__ == '__main__':
    # List of classifiers
    classifiers = (
        [nltk.DecisionTreeClassifier, []],
        [nltk.NaiveBayesClassifier, []]
        )

    # Collect data and extract features for classifiers
    print("Collecting data and extracting features... ", end='', flush=True)
    labeled_data = create_labeled_data()
    total_feature_sets = [(wsd_features(inst), label) for (inst, label) in labeled_data]
    train_set, test_set = train_test_split(total_feature_sets, shuffle=True)
    print("Done.")

    # Train classifiers
    for model in classifiers:
        print("Training %s... " % model[0].__name__, end='', flush=True)
        classifier = train_classifier(model[0], train_set, test_set=test_set, eval=False, show_d=False, show_n=False)
        model[0] = classifier

    # Train the tagger
    print("Training tagger... ", end='', flush=True)
    tagged_sents = [inst.context for (inst, label) in labeled_data]
    train_sents, test_sents = train_test_split(tagged_sents, shuffle=True)
    tagger = train_tagger(train_sents, test_sents=test_sents, eval=False)

    # Extract all sentences with the word 'interest' from one(all) of the Gutenberg corpora
    print("Preparing sentences to be classified... ", end='', flush=True)
    # all_sents = nltk.corpus.gutenberg.sents('austen-emma.txt')
    all_sents = [sent for fileid in nltk.corpus.gutenberg.fileids() for sent in nltk.corpus.gutenberg.sents(fileid)]
    untagged_sents = [sent for sent in all_sents if 'interest' in sent]
    print("Done.")
    print("%d sentences collected." % len(untagged_sents))

    # Tag the sentences and convert them into instances of SensevalInstance
    print("Tagging sentences... ", end='', flush=True)
    tagged_sentences = [tagger.tag(sent) for sent in untagged_sents]
    sents = [make_instance(sent) for sent in tagged_sentences] 
    print("Done.")

    # Classify the sentences 
    for (model, result) in classifiers:
        name = re.search(r'[^\.]+C[^\']+', str(type(model))).group()
        print("Classifying with %s... " % name, end='', flush=True)
        for sent in sents:
            result.append(run_classifier(model, sent))
        print("Done.")

    # Compare the results of the classifiers
    diff = compare_classifiers(classifiers[0], classifiers[1])



    # main program
    while True:
        print("\n%d sentences classified." % len(sents))
        print("- Enter sentence No. followed by classifier to check out result\n  (d for Decision Tree, n for Naive Bayes, eg: 17n)")
        print("- Enter c to compare the two classifiers")
        print("- Enter e to exit")
        cmd = input("What do you want to do: ").lower()

        if cmd == "e":
            break
        if cmd == "c":
            print("There are %d differences between the two classifiers." % len(diff))
            print("Sentence No.:", *[str(i+1) for i in diff])
            continue
        try:
            classifier = cmd[-1]
            idx = int(cmd[:-1])-1
            if idx < 0 or idx > len(sents):
                    raise IndexError()
            if classifier == "d":
                result = classifiers[0][1]
                name = re.search(r'[^\.]+C[^\']+', str(type(classifiers[0][0]))).group()
            elif classifier == "n":
                result = classifiers[1][1]
                name = re.search(r'[^\.]+C[^\']+', str(type(classifiers[1][0]))).group()
            else:
                raise ValueError()

            sent = sents[idx]
            tokens = [w for (w, t) in sent.context]
            print("Original sentence:")
            print(' '.join(tokens))
            tokens[sent.position] += result[idx]
            print("\nClassified with %s:" % name)
            print(' '.join(tokens))

        except:
            print("Invalid command.")
