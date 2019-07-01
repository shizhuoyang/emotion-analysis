import logging
import random

import numpy as np

from gensim.models import doc2vec
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score,roc_curve, auc
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def read_dataset():
    #dataset = pd.read_csv(path, header=0, delimiter="\t")
    #x_train, x_test, y_train, y_test = train_test_split(dataset.review, dataset.sentiment, random_state=0, test_size=0.1)
    #读取每条评论,1 代表积极情绪，0 代表消极情绪
    x_train=[]
    x_test=[]
    y_train=[]
    y_test=[]
    for line in open('pos.txt',encoding='GB2312',errors='ignore'):
        if line[:2]=='1,':
            x_train.append(line[2:])
            y_train.append(1.)
    for line in open('neg.txt'):
        if line[:2]=='0,':
            x_train.append(line[2:])
            y_train.append(0.)
    for line in open('pos_test.txt',encoding='GB2312',errors='ignore'):
        if line[:2]=='1,':
            x_test.append(line[2:])
            y_test.append(1.)
    for line in open('neg_test.txt'):
        if line[:2]=='0,':
            x_test.append(line[2:])
            y_test.append(0.)
    x_train = cleanText(x_train)
    x_test = cleanText(x_test)
    x_train = label_sentences(x_train, 'Train')
    x_test = label_sentences(x_test, 'Test')
    all_data = x_train + x_test
    return x_train, x_test, y_train, y_test, all_data

#零星的预处理
def cleanText(corpus):
    punctuation = """.,?!:;(){}[]"""
    corpus = [z.lower().replace('\n','') for z in corpus]
    corpus = [z.replace('&lt;br /&gt;', ' ') for z in corpus]
    # 将标点视为一个单词
    for c in punctuation:
        corpus = [z.replace(c, ' %s '%c) for z in corpus]
    corpus = [z.split() for z in corpus]
    return corpus

def label_sentences(corpus, label_type):
    """
    Gensim's Doc2Vec implementation requires each document/paragraph to have a label associated with it.
    We do this by using the LabeledSentence method. The format will be "TRAIN_i" or "TEST_i" where "i" is
    a dummy index of the review.
    """
    labeled = []
    for i, v in enumerate(corpus):
        label = label_type + '_' + str(i)
        labeled.append(doc2vec.TaggedDocument(v, [label]))
    return labeled


def get_vectors(doc2vec_model, corpus_size, vectors_size, vectors_type):
    """
    Get vectors from trained doc2vec model
    :param doc2vec_model: Trained Doc2Vec model
    :param corpus_size: Size of the data
    :param vectors_size: Size of the embedding vectors
    :param vectors_type: Training or Testing vectors
    :return: list of vectors
    """
    vectors = np.zeros((corpus_size, vectors_size))
    for i in range(0, corpus_size):
        prefix = vectors_type + '_' + str(i)
        vectors[i] = doc2vec_model.docvecs[prefix]
    return vectors


def train_doc2vec(corpus):
    logging.info("Building Doc2Vec vocabulary")
    d2v = doc2vec.Doc2Vec(min_count=1,  # Ignores all words with total frequency lower than this
                          window=10,  # The maximum distance between the current and predicted word within a sentence
                          vector_size=300,  # Dimensionality of the generated feature vectors
                          workers=5,  # Number of worker threads to train the model
                          alpha=0.025,  # The initial learning rate
                          min_alpha=0.00025,  # Learning rate will linearly drop to min_alpha as training progresses
                          dm=1,
                          sample=1e-3,
                          negative=5)  # dm defines the training algorithm. If dm=1 means 'distributed memory' (PV-DM)
                                 # and dm =0 means 'distributed bag of words' (PV-DBOW)
    model_dbow = doc2vec.Doc2Vec(min_count=1,  # Ignores all words with total frequency lower than this
                          window=10,  # The maximum distance between the current and predicted word within a sentence
                          vector_size=300,  # Dimensionality of the generated feature vectors
                          workers=5,  # Number of worker threads to train the model
                          alpha=0.025,  # The initial learning rate
                          min_alpha=0.00025,  # Learning rate will linearly drop to min_alpha as training progresses
                          dm=0,
                          sample=1e-3,
                          negative=5)
    d2v.build_vocab(corpus)
    model_dbow.build_vocab(corpus)
    logging.info("Training Doc2Vec model")
    # 10 epochs take around 10 minutes on my machine (i7), if you have more time/computational power make it 20
    for epoch in range(10):
        logging.info('Training iteration #{0}'.format(epoch))
        d2v.train(corpus, total_examples=d2v.corpus_count, epochs=d2v.epochs)
        model_dbow.train(corpus, total_examples=model_dbow.corpus_count, epochs=model_dbow.epochs)
        # shuffle the corpus
        random.shuffle(corpus)
        # decrease the learning rate
        d2v.alpha -= 0.0002
        model_dbow.alpha -= 0.0002
        # fix the learning rate, no decay
        d2v.min_alpha = d2v.alpha
        model_dbow.min_alpha = model_dbow.alpha

    logging.info("Saving trained Doc2Vec model")
    d2v.save("d2v.model")
    train_vectors = get_vectors(d2v, len(x_train), 300, 'Train')
    train_vectors_dbow = get_vectors(model_dbow, len(x_train), 300, 'Train')
    train_vecs=np.hstack((train_vectors,train_vectors_dbow))
    for epoch in range(10):
        logging.info('Testing iteration #{0}'.format(epoch))
        d2v.train(x_test, total_examples=d2v.corpus_count, epochs=d2v.epochs)
        model_dbow.train(x_test, total_examples=model_dbow.corpus_count, epochs=model_dbow.epochs)
        # shuffle the corpus
        random.shuffle(corpus)
        # decrease the learning rate
        d2v.alpha -= 0.0002
        model_dbow.alpha-=0.0002
        # fix the learning rate, no decay
        d2v.min_alpha = d2v.alpha
        model_dbow.min_alpha = model_dbow.alpha
    test_vectors = get_vectors(d2v, len(x_test), 300, 'Test')
    test_vectors_dbow = get_vectors(model_dbow, len(x_test), 300, 'Test')
    test_vecs=np.hstack((test_vectors,test_vectors_dbow))
    lr = SGDClassifier(loss='log', penalty='l1')
    lr.fit(train_vecs, y_train)
 
    print('Test Accuracy: %.2f'%lr.score(test_vecs, y_test))
    pred_probas = lr.predict_proba(test_vecs)[:,1]
    fpr,tpr,_ = roc_curve(y_test, pred_probas)
    print(fpr)
    print(tpr)
    """fpr=[0.         ,0.       ,  0.07692308, 0.07692308 ,0.23076923, 0.23076923,
         0.30769231, 0.30769231, 0.38461538 ,0.38461538 ,0.46153846 ,0.46153846,
         0.69230769 ,0.69230769 ,0.76923077, 0.76923077, 0.84615385 ,0.84615385,
         0.92307692 ,0.92307692 ,1.         ,1.        ]
    tpr=[0.        , 0.01666667, 0.01666667, 0.21666667, 0.21666667, 0.23333333,
         0.23333333 ,0.26666667, 0.26666667, 0.61666667, 0.61666667, 0.76666667,
         0.76666667 ,0.85 ,      0.85,       0.9 ,       0.9 ,       0.91666667,
         0.91666667 ,0.95,       0.95 ,      1.       ]"""
    roc_auc = auc(fpr,tpr)  
    plt.plot(fpr,tpr,label='area = %.2f' %roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc='lower right')
 
    plt.show()
    return d2v


'''def train_classifier(d2v, training_vectors,testing_vectors,training_labels,testing_labels):
    logging.info("Classifier training")
    train_vectors = get_vectors(d2v, len(training_vectors), 300, 'Train')
    test_vectors = get_vectors(d2v, len(testing_vectors), 300, 'Test')
    lr = SGDClassifier(loss='log', penalty='l1')
    lr.fit(train_vectors, training_labels)
    print('Test Accuracy: %.2f'%lr.score(test_vectors, testing_labels))
    test_pre=lr.predict(test_vectors)
    print(test_pre)
    #print(train_vectors)
    model = LogisticRegression()
    model.fit(train_vectors, np.array(training_labels))
    #print(training_labels)
    training_predictions = model.predict(train_vectors)
    print(training_predictions)
    logging.info('Training predicted classes: {}'.format(np.unique(training_predictions)))
    logging.info('Training accuracy: {}'.format(accuracy_score(training_labels, training_predictions)))
    logging.info('Training F1 score: {}'.format(f1_score(training_labels, training_predictions, average='weighted')))
    return model


def test_classifier(d2v, classifier, testing_vectors, testing_labels):
    logging.info("Classifier testing")
    test_vectors = get_vectors(d2v, len(testing_vectors), 300, 'Test')
    testing_predictions = classifier.predict(test_vectors)
    print(testing_labels)
    print(testing_predictions)
    logging.info('Testing predicted classes: {}'.format(np.unique(testing_predictions)))
    logging.info('Testing accuracy: {}'.format(accuracy_score(testing_labels, testing_predictions)))
    logging.info('Testing F1 score: {}'.format(f1_score(testing_labels, testing_predictions, average='weighted')))
    pred_probas = classifier.predict_proba(test_vectors)[:,1]
    fpr,tpr,_ = roc_curve(testing_labels, pred_probas)
    roc_auc = auc(fpr,tpr)
    plt.plot(fpr,tpr,label='area = %.2f' %roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc='lower right')
 
plt.show()'''

if __name__ == "__main__":
    x_train, x_test, y_train, y_test, all_data = read_dataset()
    #print(y_train)
    #print(y_test)
    d2v_model = train_doc2vec(all_data)
   # classifier = train_classifier(d2v_model, x_train,x_test, y_train,y_test)
    #test_classifier(d2v_model, classifier, x_test, y_test)
