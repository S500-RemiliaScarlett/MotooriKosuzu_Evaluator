import transformers
import keybert
import torch
from transformers import BertTokenizer, BertModel
import math
import collections
import jieba
#H函数：文本自身的价值量
#H=entropy(txt)*cos<domain_txt,domain_trg>
#余弦因子需要通过监督学习。
#学习原型机
import torch.nn as nn

class DomainClassifier(nn.Module):
    def __init__(self, num_domains):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-chinese")
        self.classifier = nn.Linear(768, num_domains)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        cls_vec = outputs.pooler_output  # [batch, 768]
        logits = self.classifier(cls_vec)  # [batch, K]
        weights = torch.sigmoid(logits)  # 
        return weights

# 推理流程
if __name__=='__main__':
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    model = DomainClassifier(num_domains=10)  # 假设10个领域
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()

    text = "特斯拉发布新款自动驾驶芯片"
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        domain_weight_vector = model(**inputs).squeeze(0).numpy()  # 长度为K的ndarray

    print(domain_weight_vector)
# 输出示例: [0.12, 0.03, 0.89, 0.01, 0.45, ...]
# 表示该文本在"科技"领域权重0.89，在"财经"领域权重0.45
###领域标签设计为控件，以便用户自主配置
#######-----------Starting-----------------####
#识文解意的爱书人明白文字中的价值。
def get_bigram_tf(word):
        # 得到二元词的词频表
        bigram_tf = {}
        for i in range(len(word) - 1):
            bigram_tf[(word[i], word[i + 1])] = bigram_tf.get(
                (word[i], word[i + 1]), 0) + 1
        return bigram_tf
def entropy(text):
    stopwords=open('stopwords.txt','r',encoding='utf-8').read().splitlines()
    words=jieba.cut(text)
    split_word=[]
    split_word_length=0
    for j in words:
        if j not in stopwords and not j.isspace():
            split_word.append(j)
            split_word_length+=1
    word_number=0
    for j in text:
        if j not in stopwords and not j.isspace():
            word_number+=1#字数统计
    word_tf=get_bigram_tf(split_word)
    blen=sum(word_tf.values())
    entropy=0
    for bg in word_tf:
        p=word_tf[bg]/blen
        entropy+=p*math.log2(p)
    return -entropy
