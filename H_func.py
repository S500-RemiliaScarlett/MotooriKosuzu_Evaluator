# -*-coding: utf-8 -*-
import transformers
import keybert
import torch
from transformers import BertTokenizer, BertModel
import math
import collections
import jieba
import os
#H函数：文本自身的价值量
#H=entropy(txt)*cos<domain_txt,domain_trg>
#余弦因子需要通过监督学习。
#学习原型机
import torch.nn as nn


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
def entropy(text,stopword=os.path.dirname(os.path.abspath(__file__))+'/stopwords.txt'):
    stopwords=open(stopword,'r',encoding='utf-8').read().splitlines()
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
    entr=0
    for bg in word_tf:
        p=word_tf[bg]/blen
        entr+=p*math.log2(p)
    return -entr
#向量分类函数作为可选项，允许开发者自行设计，默认使用torch.nn的监督学习
def H_func(text:str,req:str,classifier_function):
    entr=entropy(text)
    domain_weight_vector=classifier_function(text)
    domain_weight_req=classifier_function(req)
    # 计算余弦相似度
    dot_product = sum(a * b for a, b in zip(domain_weight_vector, domain_weight_req))
    norm_a = math.sqrt(sum(a * a for a in domain_weight_vector))
    norm_b = math.sqrt(sum(b * b for b in domain_weight_req))
    if norm_a == 0 or norm_b == 0:
        cosine_similarity = 0
    else:
        cosine_similarity = dot_product / (norm_a * norm_b)
    return entr * cosine_similarity
class domain_classifier(torch.nn.Module):
    def __init__(self, num_domains,model=os.path.dirname(os.path.abspath(__file__)) + '/models/paraphrase-multilingual-MiniLM-L12-v2'):
        super().__init__()
        self.bert = BertModel.from_pretrained(model)
        hidden_size = self.bert.config.hidden_size  # 从模型配置动态获取 hidden_size（MiniLM=384, BERT-base=768）
        self.classifier = torch.nn.Linear(hidden_size, num_domains)
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        # pooler_output 可能为 None（某些 SentenceTransformer 模型没有 pooler），
        # 此时回退到对 last_hidden_state 做 mean pooling
        cls_vec = outputs.pooler_output
        if cls_vec is None:
            cls_vec = outputs.last_hidden_state.mean(dim=1)
        logits = self.classifier(cls_vec)
        weights = torch.sigmoid(logits)
        return weights
if __name__=="__main__":
    #原型机
    '''
    #训练过程，供参考
    class TextDataset(torch.utils.data.Dataset):
        def __init__(self,texts,label):
            self.texts=texts
            self.label=label
        def __len__(self):
        # 返回数据集大小
            return len(self.texts)
        def __getitem__(self, idx):
        # 按索引返回数据和标签
            sample = self.texts[idx]
            label = self.label[idx]
            return sample, label
    #dataset load
    path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "Dataset", "THUCNews", "THUCNews")
    domains=["彩票","体育","星座","房产","家居","教育","科技","时尚","时政","游戏","娱乐","社会","股票"]
    texts=[]
    labels=[]
    for i in range(13):
        domain_path=os.path.join(path,domains[i])
        for file in os.listdir(domain_path)[0:100]:
            file_path=os.path.join(domain_path,file)
            with open(file_path,'r',encoding='utf-8') as f:
                text=f.read()
                texts.append(text)
                labels.append([0]*13)
                labels[-1][i]=1
    #数据集加载
    tokenizer = BertTokenizer.from_pretrained(os.path.dirname(os.path.abspath(__file__)) + '/models/paraphrase-multilingual-MiniLM-L12-v2')
    model = domain_classifier(num_domains=13)
    dset=TextDataset(texts, labels)
    loader = torch.utils.data.DataLoader(dset, batch_size=32, shuffle=True)
    loss_fn = torch.nn.BCELoss()  # 多标签分类损失函数（接收 sigmoid 输出，不需要内部再做 sigmoid）
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    #training
    epoch=10
    for i in range(epoch):
        for bid,batch in enumerate(loader):
            texts, labels = batch
            inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True)
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)  # forward 返回 sigmoid 权重
            # DataLoader 可能返回 list of tensors 或单个 tensor，统一处理
            if isinstance(labels, torch.Tensor):
                labels_tensor = labels.float()
            else:
                labels_tensor = torch.stack(labels).float()
            # 确保 labels 形状与 outputs 一致 [batch_size, num_domains]
            if labels_tensor.shape != outputs.shape:
                labels_tensor = labels_tensor.T
            loss = loss_fn(outputs, labels_tensor)
            loss.backward()
            optimizer.step()
            
            if(bid%10==0):
                print(f"Epoch {i+1}, Batch {bid+1}, Loss: {loss.item()}")
    torch.save(model.state_dict(), os.path.dirname(os.path.abspath(__file__))+'/model_params.pth')#保存参数
    '''
    def classifier(text):
        model=domain_classifier(num_domains=13)
        model.eval()
        tokenizer = BertTokenizer.from_pretrained(os.path.dirname(os.path.abspath(__file__)) + '/models/paraphrase-multilingual-MiniLM-L12-v2')
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        model.load_state_dict(torch.load(os.path.dirname(os.path.abspath(__file__))+'/model_params.pth', map_location=torch.device('cpu')))
        with torch.no_grad():
            domain_weight_vector = model(inputs['input_ids'], inputs['attention_mask']).squeeze(0).numpy()  # 长度为K的ndarray
        return domain_weight_vector
    text="科技发展日新月异，人工智能正在改变我们的生活。"
    req="科技"
    H_func_value=H_func(text,req,classifier)
    print(f"H函数值: {H_func_value}")