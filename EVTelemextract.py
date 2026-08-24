# -*- coding: utf-8 -*-
from transformers import BertTokenizer, BertModel
import jieba
import torch
import os
import sys
import numpy as np
MODEL_PATH="models/paraphrase-multilingual-MiniLM-L12-v2"
'''
这些模型用以提取事件要素，分别是时间、地点、人物、事件、原因、结果
'''
class eventelement:
    def __init__(self,text,elements):
        self.text = text
        self.elements=elements
    def __len__(self):
            return len(self.texts)
    def __getitem__(self, idx):
            return self.texts[idx], self.elements[idx]
#7要素使用不同的模型实现
class sitepredict(torch.nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_path)
        hidden_size = self.bert.config.hidden_size  # 从模型配置动态获取 hidden_size（MiniLM=384, BERT-base=768）
        self.linear1=torch.nn.Linear(hidden_size, hidden_size)
        self.start_site_pr = torch.nn.Linear(hidden_size, 1)  
        self.end_site_pr=torch.nn.Linear(hidden_size,1)
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        cls_vec = outputs.pooler_output
        if cls_vec is None:
            mask = attention_mask.unsqueeze(-1).to(outputs.last_hidden_state.dtype)
            cls_vec = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        cls_vec = self.linear1(cls_vec)
        cls_vec=torch.leaky_relu(cls_vec, negative_slope=0.01)
        start_site_probs = torch.sigmoid(self.start_site_pr(cls_vec))
        end_site_probs = torch.sigmoid(self.end_site_pr(cls_vec))
        return start_site_probs, end_site_probs
#When,who,where
when=sitepredict(os.path.join(os.path.dirname(__file__), MODEL_PATH))
who=sitepredict(os.path.join(os.path.dirname(__file__), MODEL_PATH))
where=sitepredict(os.path.join(os.path.dirname(__file__), MODEL_PATH))
#what采用生成式
class whatpredict(torch.nn.Module):
    def __init__(self, model_path,hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bert = BertModel.from_pretrained(model_path)
        hidden_size = self.bert.config.hidden_size
        self.gru = torch.nn.GRU(hidden_size, hidden_dim)
        self.what_pr = torch.nn.Linear(hidden_dim, hidden_size)
    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_dim)
    def forward(self,input_ids,attention_mask,h):
        bert_mod=self.bert(input_ids,attention_mask=attention_mask)
        cls_vec=bert_mod.pooler_output
        if cls_vec is None:
            mask = attention_mask.unsqueeze(-1).to(outputs.last_hidden_state.dtype)
            cls_vec = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        cls_vec,hidden= self.gru(cls_vec,h)
        cls_vec = self.what_pr(cls_vec.view(-1, self.hidden_dim))
        output_vec=torch.softmax(cls_vec,dim=1)
        return output_vec,hidden
    def generate(self,input_ids,attention_mask,hidden):
        bert_mod=self.bert(input_ids,attention_mask=attention_mask)
        cls_vec=bert_mod.pooler_output
        if cls_vec is None:
            mask = attention_mask.unsqueeze(-1).to(outputs.last_hidden_state.dtype)
            cls_vec = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

class judge(torch.nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_path)
        hidden_size = self.bert.config.hidden_size
        self.linear1 = torch.nn.Linear(hidden_size, hidden_size)
        self.what_pr = torch.nn.Linear(hidden_size, 1)
    def forward(self,input_ids,attention_mask):
        bert_mod=self.bert(input_ids,attention_mask=attention_mask)
        cls_vec=bert_mod.pooler_output
        if cls_vec is None:
            mask = attention_mask.unsqueeze(-1).to(outputs.last_hidden_state.dtype)
            cls_vec = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        cls_vec = self.linear1(cls_vec)
        cls_vec=torch.leaky_relu(cls_vec, negative_slope=0.01)
        output_vec=torch.sigmoid(cls_vec,dim=1)
        return output_vec

#why,eff
class sentence_dataset:
    def __init__(self, texts, types):
        self.texts = texts
        self.types = types
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        return self.texts[idx], self.types[idx]
class reasoneffpredict(torch.nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_path)
        hidden_size = self.bert.config.hidden_size
        self.linear1 = torch.nn.Linear(hidden_size, hidden_size)
        self.reason_pr = torch.nn.Linear(hidden_size, 1)
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        cls_vec = outputs.pooler_output
        if cls_vec is None:
            mask = attention_mask.unsqueeze(-1).to(outputs.last_hidden_state.dtype)
            cls_vec = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        cls_vec = self.linear1(cls_vec)
        cls_vec=torch.leaky_relu(cls_vec, negative_slope=0.01)
        reason_probs = torch.sigmoid(self.reason_pr(cls_vec))
        return reason_probs
