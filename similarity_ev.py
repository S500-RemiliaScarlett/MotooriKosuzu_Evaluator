#!/usr/bin/python3
# -*-coding: utf-8 -*-
import hashlib
import transformer
import keybert
def kw_extract(text,Model='models\paraphrase-multilingual-MiniLM-L12-v2',topk=5,stop_words='chinese',keyphrase_ngram_range=(1, 3)):
    '''
    keybert可以加载模型
    '''
    kb=keybert.KeyBERT(model=Model)
    keywords=kb.extract_keywords(text,top_n=topk,stop_words=stop_words,keyphrase_ngram_range=keyphrase_ngram_range)
    return keywords
def simhash(s1,s2):
    '''
    计算两个文本的simhash相似度
    汉明距离：对两个二进制数进行异或运算，统计结果中1的个数
    由于最后是根据两个表的正、负确定0和1，所以单个位异或为1等价为表对应位置数相乘为负数
    汉明距离越大，文本相似性越低
    '''
    kw_set1=list([i for i in kw_extract(s1)])
    kw_set2=list([i for i in kw_extract(s2)])
    hs_list1=[hashlib.sha1(i[0].encode('utf-8')).hexdigest() for i in kw_set1]
    hs_list2=[hashlib.sha1(i[0].encode('utf-8')).hexdigest() for i in kw_set2]
    hs_list1=[bin(int(i,16)).replace('0b','') for i in hs_list1]
    hs_list2=[bin(int(i,16)).replace('0b','') for i in hs_list2]
    x_list1=[]
    x_list2=[]
    for i in range(len(hs_list1)):
        x_list1.append([kw_set1[i][1] if j=='1' else -kw_set1[i][1] for j in hs_list1[i]])
    for i in range(len(hs_list2)):
        x_list2.append([kw_set2[i][1] if j=='1' else -kw_set2[i][1] for j in hs_list2[i]])
    s1_list=[sum(i) for i in zip(*x_list1)]
    s2_list=[sum(i) for i in zip(*x_list2)]
    sim=0
    for i in range(len(s1_list)):
        if s1_list[i]*s2_list[i]>0:
            sim+=1
    return sim/len(s1_list)