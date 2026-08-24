from transformers import BertTokenizer, BertModel
import jieba
import torch
import os
import sys
import numpy as np
#分类器模型
class Classifier:
    def __init__(self, num_domains,model):
        super().__init__()
        self.bert = BertModel.from_pretrained(model)
        hidden_size = self.bert.config.hidden_size  # 从模型配置动态获取 hidden_size（MiniLM=384, BERT-base=768）
        self.classifier = torch.nn.Linear(hidden_size, num_domains)
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        # pooler_output 可能为 None（某些 SentenceTransformer 模型没有 pooler），
        # 此时回退到对 last_hidden_state 做 mean pooling。
        # 注意：混合训练会把短文本和长文本放进同一个 batch，若不屏蔽 padding，
        # 短文本会被大量 [PAD] 向量稀释，重新塌缩成「平行于全1向量」的平坦输出。
        cls_vec = outputs.pooler_output
        if cls_vec is None:
            mask = attention_mask.unsqueeze(-1).to(outputs.last_hidden_state.dtype)
            cls_vec = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        logits = self.classifier(cls_vec)
        weights = torch.sigmoid(logits)  # 多标签输出：每个领域独立 0~1，不强制和为1
        return weights
