# Multilayer Percpetron Classifier to find SPAM messages
__author__ = "Chirayu Desai"

import csv
import operator
import warnings
import collections
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings('ignore')
# read data from csv
df = pd.read_csv('labelled_input.csv', names=['sms', 'class'], encoding='ansi')

x = df['sms']
y = df['class']

# 10 fold cross validation
kf = KFold(n_splits=10)

# available activation function types
activation_types = ['identity', 'logistic', 'tanh', 'relu']
performance = dict()


def parse_classification_report(clf_report):
    """
    Source StackOverflow
    Parse a sklearn classification report into a dict keyed by class name
    and containing a tuple (precision, recall, fscore, support) for each class
    """
    lines = clf_report.split('\n')
    # Remove empty lines
    lines = list(filter(lambda l: not len(l.strip()) == 0, lines))

    # Starts with a header, then score for each class and finally an average
    header = lines[0]
    cls_lines = lines[1:-1]
    avg_line = lines[-1]

    assert header.split() == ['precision', 'recall', 'f1-score', 'support']
    assert avg_line.split()[1] == 'avg'

    # We cannot simply use split because class names can have spaces. So instead
    # figure the width of the class field by looking at the indentation of the
    # precision header
    cls_field_width = len(header) - len(header.lstrip())

    # Now, collect all the class names and score in a dict

    def parse_line(l):
        """Parse a line of classification_report"""
        class_name = l[:cls_field_width].strip()
        precision, recall, f_score, support = l[cls_field_width:].split()
        precision = float(precision)
        recall = float(recall)
        f_score = float(f_score)
        support = int(support)
        return class_name, precision, recall, f_score, support

    data = collections.OrderedDict()
    for line in cls_lines:
        if 'accuracy' not in line:
            ret = parse_line(line)
            cls_name = ret[0]
            scores = ret[1:]
            data[cls_name] = scores

    # average
    data['avg'] = parse_line(avg_line)[1:]
    return data


def run_classifier(activation_type, test_flag):
    """
    Runs a classifier on given activation type
    :param activation_type: given activation type
    :param test_flag: if True , runs the classifier on 300 sms Test data
    """
    # 6 nodes for 1st Hidden Layer, 10 nodes for 2st Hidden Layer, maximum of 1000 iterations
    classifier = MLPClassifier(activation=activation_type, batch_size='auto',
                               hidden_layer_sizes=(6, 10), learning_rate='constant', max_iter=1000,
                               random_state=None, solver='adam')  #

    print("Using " + classifier.activation + " activation")
    # vectorize the inputs
    vectorizer = TfidfVectorizer()

    ind = 0
    precision_dict = dict()
    recall_dict = dict()
    for train_indices, test_indices in kf.split(x, y):
        x_train, x_test, y_train, y_test = x[train_indices], x[test_indices], y[train_indices], y[test_indices]
        train_x_vector = vectorizer.fit_transform(x_train)
        test_x_vector = vectorizer.transform(x_test)
        classifier.fit(train_x_vector, y_train)
        guess = classifier.predict(test_x_vector)
        rep = classification_report(y_test, guess)
        a, b, c, d = dict(parse_classification_report(rep))['avg']
        precision_dict[ind] = a
        recall_dict[ind] = b
        ind = ind + 1

    p, r = (float(sum(precision_dict.values())) / 10), (float(sum(recall_dict.values())) / 10)
    performance[act_type] = (p+r)*50
    print("Precision : ", p * 100, ",  Recall : ", r * 100)
    if test_flag:
        td = pd.read_csv('test_sms.csv', names=['sms'], encoding='ansi')
        # print(td)
        op_x = td['sms']
        op_x_vector = vectorizer.transform(op_x)
        prediction = classifier.predict(op_x_vector)
        # for i in range(0, len(prediction)):
        #     print(op_x[i] + " -> " + prediction[i])

        with open('output_labelled1.csv', 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            for i in range(0, len(prediction)):
                writer.writerow([op_x[i], prediction[i]])
        with open('labels.txt', 'w') as result_file:
            for i in range(0, len(prediction)):
                result_file.write(prediction[i]+"\n")

        print("Prediction Complete")


# run for all activation types
for act_type in activation_types:
    run_classifier(act_type, False)

# pick the best activation
best = max(performance.items(), key=operator.itemgetter(1))[0]
print(best, " seems to perform the best.")

print("Now Predicting with ", best)
# predict with the best activation found
run_classifier(best, True)
