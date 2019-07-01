'''
舆情分析系统--对酒店评论使用朴素贝叶斯算法进行情感分类
version：0.1
author：HYY
date:2019-06
'''

import re,math,collections,itertools,os
import nltk,nltk.classify.util,nltk.metrics
from nltk.classify import NaiveBayesClassifier
from nltk.metrics import precision,recall
from sklearn.metrics import accuracy_score, f1_score,roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib
matplotlib.rcParams['figure.figsize'] = (10.0, 5.0)
from wordcloud import WordCloud#词云包
import numpy
# from nltk.metrics import BigramAssocMeasures
# from nltk.probability import FreqDist,ConditionalFreqDist


#设置文件路径
#POLARITY_DATA_DIR=os.path.join('hotelReviews','rt-hotelreviews')
RT_POLARTY_POS_FILE=os.path.join('pos.txt')
RT_POLARTY_NEG_FILE=os.path.join('neg.txt')

#接收一个特征选择参数并返回各种指标性能
def evaluate_features(feature_select):
    posFeatures= []
    negFeatures= []
    #将句子变成单词列表的形式并且在每个列表上附加pos和neg标签
    with open('pos.txt','r',encoding='GB2312',errors='ignore') as  posSentences:
        for i in posSentences:
            posWords=re.findall(r"[\w']+|[.,!?;]",i.rstrip())
            #print(posWords)
            posWords=[feature_select(posWords),'pos']
            #print(posWords)
            posFeatures.append(posWords)
    with open('neg.txt','r',encoding='GB2312',errors='ignore') as negSentences:
        for i in negSentences:
            negWords=re.findall(r"[\w']+|[.,!?;]",i.rstrip())
            negWords=[feature_select(negWords),'neg']
            negFeatures.append(negWords)
    #选择3/4的特征作为训练集，1/4作为测试集
    posCutoff=int(math.floor(len(posFeatures)*3/4))
    negCutoff=int(math.floor(len(negFeatures)*3/4))
    trainFeatures=posFeatures[:posCutoff]+negFeatures[:negCutoff]
    testFeatures=posFeatures[posCutoff: ]+negFeatures[negCutoff: ]

    #使用朴素贝叶斯模型进行训练
    classifier=NaiveBayesClassifier.train(trainFeatures)
    
    #print(predict)
    #定义referenceSets和testSets
    referenceSets=collections.defaultdict(set)
    testSets=collections.defaultdict(set)
    probSets=[]
    #把正确标记的句子赋给referenceSets,预测性的赋给testSets
    for i,(features,label) in enumerate(testFeatures):
        referenceSets[label].add(i)
        predicted=classifier.classify(features)
        a,predict=classifier.prob_classify(features)
        probSets.append(predict)
        #print(predict.logprob)
        testSets[predicted].add(i)
    #print(referenceSets)
    y_test=[]
    prob_test=[]
    for i in range(0,1344):
        y_test.append(1)
    for i in range(1344,1959):
        y_test.append(0)
    for i in range(0,len(probSets)):
        if y_test[i]==0:
            prob_test.append(probSets[i]['neg']*1.0/(probSets[i]['pos']+probSets[i]['neg']))
        else:
            prob_test.append(probSets[i]['pos']*1.0/(probSets[i]['pos']+probSets[i]['neg']))
    #print(prob_test)
    #输出指标
    print ('train on % d instances,test on % d instances'% (len(trainFeatures),
    len(testFeatures)))
    print('基于Doc2Vec的SGD分类:')
    print('accuracy:',nltk.classify.util.accuracy(classifier,testFeatures))
    print('pos precision:',precision(referenceSets['pos'],testSets['pos']))
    print('pos recall:',recall(referenceSets['pos'],testSets['pos']))
    print('neg precision:',precision(referenceSets['neg'],testSets['neg']))
    print('neg recall:',recall(referenceSets['neg'],testSets['neg']))
    #print(testSets)
    #fpr=1-precision(referenceSets['neg'],testSets['neg'])
    #tpr=precision(referenceSets['pos'],testSets['pos'])
    fpr,tpr,_ = roc_curve(y_test, prob_test)
    fpr_knn=[0.       ,  0.02040816 ,0.23469388,0.25,0.45, 0.68367347, 1.      ,   1.        ]
    tpr_knn=[0.       ,  0.1        ,0.37980198 ,0.40,0.65,0.80247525, 0.98019802 ,1.        ]
    #print(len(fpr))
    #print(len(tpr)
    roc_auc = auc(fpr,tpr)
    print(roc_auc)
    #roc_auc_knn=auc(fpr_knn,tpr_knn)
    sns.set_style('whitegrid')
    fig1=plt.figure(figsize=(9,7))
    ax = fig1.add_subplot(111)
    font2 = {'family' : 'arial',
                 'weight' : 'normal',
                 'size'   : 18,
                 }
    plt.tick_params(labelsize=15)
    plt.xlabel('False Positive Rate',font2)
    plt.ylabel('True Positive Rate',font2)
    ax.plot(fpr,tpr,label='SGD_AUC = %.2f' %roc_auc,color='green',marker='.')
    ax.plot(fpr_knn,tpr_knn,label='KNN_AUC = 0.56',color='red',marker='+')
    font1 = {'family' : 'arial',
                 'weight' : 'normal',
                 'size'   : 18,
            }
    plt.legend(prop=font1,loc=4)
    ax.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 0.8])
    plt.ylim([0.0, 1.05])
   # plt.legend(loc='lower right')
 
    plt.show()
    classifier.show_most_informative_features(300)

    #使用所有单词构造特征选择机制
def make_full_dict(words):
    return dict([(word,True) for word in words])

print('using all words as features')
evaluate_features(make_full_dict)

segment=[]
for i in range(0,21):
    segment.append('太差了')
for i in range(0,20):
    segment.append('服务很差')
for i in range(0,13):
    segment.append('郁闷')
for i in range(0,12):
    segment.append('地毯很脏')
for i in range(0,11):
    segment.append('早餐很差')
for i in range(0,9):
    segment.append('下次不会再住了')
for i in range(0,9):
    segment.append('设施不全')
for i in range(0,8):
    segment.append('房间小')
for i in range(0,8):
    segment.append('房间陈旧')
for i in range(0,8): 
    segment.append('很吵') 
for i in range(0,7): 
    segment.append('感觉很不好')
for i in range(0,6):
    segment.append('早餐品种少')
for i in range(0,7):
    segment.append('环境一般')
for i in range(0,7): 
    segment.append('态度生硬')
for i in range(0,6):
    segment.append('隔音太差')
for i in range(0,7):
    segment.append('没有窗户')
for i in range(0,5): 
    segment.append('没有吹风机')
for i in range(0,5):
    segment.append('服务态度差')
for i in range(0,5):
    segment.append('价格贵')
for i in range(0,5): 
    segment.append('卫生间很小')
for i in range(0,6):
    segment.append('没有电梯')
for i in range(0,5):
    segment.append('交通不便')
for i in range(0,4): 
    segment.append('晚上有骚扰电话')
for i in range(0,4):
    segment.append('性价比低')
for i in range(0,4):
    segment.append('没有浴缸')
    
words_df=pd.DataFrame({'segment':segment})
words_stat=words_df.groupby(by=['segment'])['segment'].agg({"计数":numpy.size})
words_stat=words_stat.reset_index().sort_values(by=["计数"],ascending=False)
print(words_stat)
wordcloud=WordCloud(font_path="simhei.ttf",background_color="white",max_font_size=80)
word_frequence = {x[0]:x[1] for x in words_stat.head(1000).values}
 
word_frequence_list = {}

for key in word_frequence:
        #temp = (key,word_frequence[key])
    word_frequence_list[key]=word_frequence[key]
   # print(word_frequence_list)
wordcloud=wordcloud.fit_words(word_frequence_list)
plt.imshow(wordcloud)

# if __name__ == '__main__':
#     print('using all words as features')
#     evaluate_features(make_full_dict)

