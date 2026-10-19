# -*-coding: utf-8 -*-
#欢迎使用"MarisaForge"
import torch
from transformers import BertTokenizer, BertModel
import math
import jieba
import jieba.analyse as analyse
import os
import sys
import sentence_transformers
import json
#监督学习
class classifiermodel(torch.nn.Module):
    def __init__(self,model_path,output_dimension,middle_layers:list):
        super().__init__()
        self.tokenizer=BertTokenizer.from_pretrained(model_path)
        self.bertmodel=BertModel.from_pretrained(model_path)
        self.bertmodel.eval()
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        BThidden=self.bertmodel.config.hidden_size#Bert模型的隐藏层维度
        # 关键修复1：中间层必须用 nn.ModuleList 注册。
        # 原来是普通 Python list，这些层不在 parameters()/state_dict() 里，
        # 因此既不会被优化器训练，也不会被 torch.save 保存，更不会被 .to(device) 迁移。
        # 结果只有最后的 output 层在训练（BERT 冻结、中间层随机初始化冻结），
        # 模型在随机特征上做线性拟合，损失在 0.63 附近振荡就再也下不去。
        self.middle=torch.nn.ModuleList([torch.nn.Linear(BThidden,middle_layers[0])])
        for i in range(1,len(middle_layers)):
            self.middle.append(torch.nn.Linear(middle_layers[i-1],middle_layers[i]))
        self.output=torch.nn.Linear(middle_layers[-1],output_dimension)
        self.to(self.device)#整机迁移（含 middle 与 output），原来只迁移了 bert
    def forward(self,text):
        token=self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        input_ids=token['input_ids'].to(self.device)
        attention_mask=token['attention_mask'].to(self.device)
        hidden=self.bertmodel(input_ids, attention_mask=attention_mask).last_hidden_state  # [batch, seq_len, hidden]
        # mean pooling：MiniLM 是 sentence-transformer，训练时用的是均值池化而非 CLS token（[:,0]）。
        # 用 CLS 做细粒度主题分类特征不够强；这里屏蔽 padding 段后对 token 向量取平均。
        mask=attention_mask.unsqueeze(-1).float()  # [batch, seq_len, 1]
        input_vec=(hidden*mask).sum(dim=1)/mask.sum(dim=1).clamp(min=1e-9)  # [batch, hidden]
        LRelu=torch.nn.LeakyReLU(negative_slope=0.01)
        for layer in self.middle:
            input_vec=layer(input_vec)
            input_vec=LRelu(input_vec)

        output=self.output(input_vec)
        output=torch.sigmoid(output)
        return output
class dataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
class classifierfunction:
    def __init__(self,model_path:str,output_dimension:int,middle_layers:list,single_domain:bool):
        self.model=classifiermodel(model_path,output_dimension,middle_layers)
        if(single_domain):
            self.judgemodel=classifiermodel(model_path,1,middle_layers)
        self.single_domain=single_domain
        self.dataset=None
        self.judgeset=None
    def load_dataset(self,file_path:str,data_keywords:list=[],labels_keywords:list=[],sharpen_labels:bool=True):
        data_keywords+=["text"]
        labels_keywords+=["label"]
        with open(file_path,encoding="utf-8") as f:
            data=json.load(f)
            try:
                data_list=data["data"]
            except KeyError:
                print("数据集格式不正确，请使用data字段存储数据")
                return
            texts=[]
            labels=[]
            isnegative=[]
            for item in data_list:
                for key in data_keywords:
                    if key in item:
                        texts.append(item[key])
                        break
                for key in labels_keywords:
                    if key in item:
                        labels.append(item[key])
                        if(len(labels[-1])<self.model.output.weight.shape[0]):
                            labels[-1]=labels[-1]+[0]*(self.model.output.weight.shape[0]-len(labels[-1]))#零补齐
                        if(len(labels[-1])>self.model.output.weight.shape[0]):
                            labels[-1]=labels[-1][0:self.model.output.weight.shape[0]]#硬截断
                        break
            # 标签锐化（可选，默认开）。仅当 label 是「和为1的主题比例」这类软标签时才需要：
            # 软标签用 sigmoid+BCE 训练会有熵下限（损失最低 ~0.45），输出会停在平坦的边际均值。
            # 锐化成硬 one-hot 后 BCE 才能学出尖锐峰值。若你的标签本来就是硬 one-hot，本步幂等；
            # 若是多标签（一个样本可同时命中多个类，如 [1,1,0]），请传 sharpen_labels=False，否则会破坏多标签语义。
            if sharpen_labels:
                for li in range(len(labels)):
                    lab=labels[li]
                    m=max(lab)
                    if m>0:
                        idx=lab.index(m)#严格取第一个最大维度，避免多维度并列时产出 [1,1,..] 的平坦目标
                        labels[li]=[1.0 if j==idx else 0.0 for j in range(len(lab))]
                    else:
                        labels[li]=[0.0 for _ in lab]
            if self.single_domain:
                # 过滤器数据集：全部样本（含域外），标签是 is_negative 的二值 0/1
                for item in data_list:
                    isnegative.append([1.0 if item.get("is_negative", False) else 0.0])
                isnegative=torch.tensor(isnegative)
                self.judgeset=dataset(texts,isnegative)
                # 主题分类数据集：剔除 is_negative=True 的域外样本，避免它们污染主题标签训练。
                # 依赖 texts/labels/data_list 三者按下标一一对应（每条样本都含 text 与 label）。
                topic_texts=[]; topic_labels=[]
                for i,item in enumerate(data_list):
                    if item.get("is_negative", False):
                        continue
                    topic_texts.append(texts[i]); topic_labels.append(labels[i])
                self.dataset=dataset(topic_texts,topic_labels)
            else:
                isnegative=torch.tensor([])  # 未使用，仅占位
                self.dataset=dataset(texts,labels)
    def train_classifier(self,epochs:int,batch_size:int,learning_rate:float,param_path:str,pre_trained_params=None):
        """训练主题标签分类器 self.model（只吃 self.dataset 里的域内样本）。"""
        if(pre_trained_params!=None):
            self.model.load_state_dict(torch.load(pre_trained_params))
        self.model.train()
        self.model.bertmodel.eval()
        self.model.bertmodel.requires_grad_(False)
        dset=torch.utils.data.DataLoader(self.dataset,batch_size=batch_size,shuffle=True)
        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()),
                                  lr=learning_rate, weight_decay=1e-4)#weight_decay是L2惩罚因子；1e-2过大，会持续把权重拉向0，令sigmoid输出平坦在0.5
        for i in range(epochs):
            for bid,batch in enumerate(dset):
                texts,labels=batch
                optimizer.zero_grad()
                outputs=self.model(texts)
                if isinstance(labels, torch.Tensor):
                    labels_tensor = labels.float()
                else:
                    labels_tensor = torch.stack(labels).float()
                if labels_tensor.shape != outputs.shape:
                    labels_tensor = labels_tensor.T
                loss=loss_fn(outputs,labels_tensor)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # 防梯度爆炸
                optimizer.step()
                if(bid%10==0):
                    print("轮次:%d\t样本数:%d\t损失:%.5lf\t"%(i+1,bid+1,loss.item()))
        print("正在保存主题分类器参数")
        torch.save(self.model.state_dict(),param_path)
    def train_filter(self,epochs:int,batch_size:int,learning_rate:float,filter_path:str,pre_trained_params=None):
        """训练过滤器 self.judgemodel（吃全部样本，二值 is_negative 标签）。"""
        if not self.single_domain:
            print("single_domain=False，无过滤器可训练")
            return
        if(pre_trained_params!=None):
            self.judgemodel.load_state_dict(torch.load(pre_trained_params))
        self.judgemodel.train()
        self.judgemodel.bertmodel.eval()
        self.judgemodel.bertmodel.requires_grad_(False)
        jset=torch.utils.data.DataLoader(self.judgeset,batch_size=batch_size,shuffle=True)
        loss_fn = torch.nn.BCELoss()
        # 关键修复2：过滤器是独立的 judgemodel，必须单独用一个优化器训练它。
        # 原来只用 self.model 的参数建优化器，judgemodel 的梯度算了但从不更新，
        # 过滤器损失永远停在随机基线 ln2≈0.693（即你看到的 0.70 附近）。
        # 注意不要用一个共享优化器同时包两个模型：AdamW 每一步都会对参数施加 weight_decay，
        # 哪怕该步梯度为 0，会导致两个模型在对方训练时被白白衰减。
        optimizer=torch.optim.AdamW(filter(lambda p: p.requires_grad, self.judgemodel.parameters()),
                                    lr=learning_rate, weight_decay=1e-4)
        for i in range(epochs):
            for bid,batch in enumerate(jset):
                texts,labels=batch
                optimizer.zero_grad()
                outputs=self.judgemodel(texts)
                if isinstance(labels, torch.Tensor):
                    labels_tensor = labels.float()
                else:
                    labels_tensor = torch.stack(labels).float()
                if labels_tensor.shape != outputs.shape:
                    labels_tensor = labels_tensor.T
                loss=loss_fn(outputs,labels_tensor)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.judgemodel.parameters(), 1.0)  # 防梯度爆炸
                optimizer.step()
                if(bid%10==0):
                    print("过滤器轮次:%d\t样本数:%d\t损失:%.5lf\t"%(i+1,bid+1,loss.item()))
        print("正在保存过滤器参数")
        torch.save(self.judgemodel.state_dict(),filter_path)
    def train(self,train_dataset:dataset,epochs:int,batch_size:int,learning_rate:float,param_path:str,pre_trained_params=None):
        """兼容旧接口：一次性训练主题分类器与过滤器。也可分别调用 train_classifier / train_filter。"""
        self.train_classifier(epochs,batch_size,learning_rate,param_path,pre_trained_params)
        if self.single_domain:
            self.train_filter(epochs,batch_size,learning_rate,os.path.dirname(param_path)+"/filterparam.pth")
    def classifier(self,text,param_path,filter_path):
        self.model.eval()
        with torch.no_grad():
            self.model.load_state_dict(torch.load(param_path))
            if(self.single_domain):
                self.judgemodel.eval()
                self.judgemodel.load_state_dict(torch.load(filter_path))
                filter_output=self.judgemodel(text)
                if(filter_output>0.5):
                    return torch.zeros(self.model.output.weight.shape[0])
            output=self.model(text)
            return output
    def test(self,text,param_path,filter_path):
        self.model.eval()
        with torch.no_grad():
            self.model.load_state_dict(torch.load(param_path))
            if(self.single_domain):
                self.judgemodel.eval()
                self.judgemodel.load_state_dict(torch.load(filter_path))
                filter_output=self.judgemodel(text)
                print("过滤器输出:",filter_output)
            output=self.model(text)
            print("模型输出:",output)
            if(self.single_domain):
                return output,filter_output
            return output
# ---------------------------------------------------------------------------
# 下层 TWS 领域本体：六大实体抽取（通用事件要素抽取模型）
# ---------------------------------------------------------------------------
# 依据《TWS 领域本体设计说明书》§4.1 / §4.2 的六大实体：
#   产品型号 product_model / 技术方案 tech_solution / 功能点 feature /
#   使用场景 usage_scenario / 质量问题 quality_issue / 用户群体 user_group
# 模型参考 EVTelemextract.py 的多头 BERT 思路：共享编码器 + 每个实体类型一个
# 「存在性判断 + 片段定位」头，直接从原文抽出实体名，返回结构化实体信息。
SIX_ENTITY_TYPES = ['product_model', 'tech_solution', 'feature',
                    'usage_scenario', 'quality_issue', 'user_group']

# 各实体类型的属性 Schema（§4.2），用于返回实体信息时对齐字段。
# name_field 为该类型的规范名字段；attributes 为完整属性列表；required 为必填项。
ENTITY_TYPE_INFO = {
    'product_model': {
        'name': '产品型号', 'dimension': '产品形态', 'name_field': 'model_name',
        'attributes': ['brand', 'model_name', 'alias', 'price', 'release_date', 'wear_form', 'category'],
        'required': ['brand', 'model_name', 'wear_form'],
    },
    'tech_solution': {
        'name': '技术方案', 'dimension': '技术演化', 'name_field': 'tech_name',
        'attributes': ['tech_name', 'tech_alias', 'tech_type', 'vendor', 'generation', 'maturity'],
        'required': ['tech_name', 'tech_type'],
    },
    'feature': {
        'name': '功能点', 'dimension': '产品形态', 'name_field': 'feature_name',
        'attributes': ['feature_name', 'metric', 'experience', 'feature_type'],
        'required': ['feature_name', 'feature_type'],
    },
    'usage_scenario': {
        'name': '使用场景', 'dimension': '用户喜好', 'name_field': 'scenario_name',
        'attributes': ['scenario_name', 'demand', 'related_features', 'target_groups'],
        'required': ['scenario_name'],
    },
    'quality_issue': {
        'name': '质量问题', 'dimension': '网络舆论', 'name_field': 'issue_name',
        'attributes': ['issue_name', 'issue_category', 'severity', 'scope', 'related_entities'],
        'required': ['issue_name', 'issue_category'],
    },
    'user_group': {
        'name': '用户群体', 'dimension': '用户喜好', 'name_field': 'group_name',
        'attributes': ['group_name', 'traits', 'preferences', 'pain_points'],
        'required': ['group_name'],
    },
}


class domainentitymodel(torch.nn.Module):
    """六大领域实体抽取模型（参考 EVTelemextract.py 的多头 BERT 思路）。

    共享 BERT 编码器，6 个实体头各含三路输出：
      presence —— 该实体类型是否出现在文本中（raw logits，训练用 BCEWithLogits）；
      start    —— 每个 token 作为实体名起始位置的 logits；
      end      —— 每个 token 作为实体名结束位置的 logits。

    与 EVTelemextract.py 的区别：把其中的「pooled 向量回归」改成标准 token 级
    片段定位（span extraction），从而真正从原文抽出实体名。
    """

    def __init__(self, model_path):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.bert = BertModel.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hidden = self.bert.config.hidden_size
        self.presence = torch.nn.ModuleDict({
            et: torch.nn.Linear(hidden, 1) for et in SIX_ENTITY_TYPES
        })
        self.start = torch.nn.ModuleDict({
            et: torch.nn.Linear(hidden, 1) for et in SIX_ENTITY_TYPES
        })
        self.end = torch.nn.ModuleDict({
            et: torch.nn.Linear(hidden, 1) for et in SIX_ENTITY_TYPES
        })
        self.to(self.device)

    def _encode(self, text):
        """编码文本，返回 (input_ids, attention_mask, last_hidden_state, pooled)。"""
        enc = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        input_ids = enc['input_ids'].to(self.device)
        attention_mask = enc['attention_mask'].to(self.device)
        hidden = self.bert(input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return input_ids, attention_mask, hidden, pooled

    def forward(self, text):
        """返回 (presence, start, end, input_ids, attention_mask)。

        presence: dict[etype] -> [B] raw logits
        start/end: dict[etype] -> [B, L] raw logits（未归一化）
        """
        input_ids, attention_mask, hidden, pooled = self._encode(text)
        presence = {et: self.presence[et](pooled).squeeze(-1)
                    for et in SIX_ENTITY_TYPES}
        start = {et: self.start[et](hidden).squeeze(-1) for et in SIX_ENTITY_TYPES}
        end = {et: self.end[et](hidden).squeeze(-1) for et in SIX_ENTITY_TYPES}
        return presence, start, end, input_ids, attention_mask

    def extract(self, text, presence_threshold=0.5):
        """抽取单条文本的六大实体，返回结构化实体信息。

        返回 dict[etype] -> {
            'present': bool, 'name': str|None, 'confidence': float,
            'start'/'end': token 下标（仅在 present 时存在）,
            'attributes': {attr: None}  # 属性留待映射层填充（§8 接口约定）
        }
        实体名按字符 offset 从原文精确截取，避免 subword/OOV 造成的拼接与空格问题。
        """
        self.eval()
        with torch.no_grad():
            presence, start, end, input_ids, attention_mask = self.forward(text)
        enc = self.tokenizer(text, return_tensors='pt', return_offsets_mapping=True,
                             truncation=True)
        offsets = enc['offset_mapping'][0].tolist()  # [(c0, c1), ...]
        result = {}
        for et in SIX_ENTITY_TYPES:
            p = float(torch.sigmoid(presence[et][0]).item())
            info = {
                'present': p >= presence_threshold,
                'name': None,
                'confidence': p,
                'attributes': {attr: None for attr in ENTITY_TYPE_INFO[et]['attributes']},
            }
            if info['present']:
                mask = attention_mask[0].float()
                neg = (1.0 - mask) * -1e9
                s_prob = torch.softmax(start[et][0] + neg, dim=0)
                e_prob = torch.softmax(end[et][0] + neg, dim=0)
                si = int(s_prob.argmax().item())
                ei = int(e_prob.argmax().item())
                if ei < si:
                    si, ei = ei, si
                info['start'], info['end'] = si, ei
                cs = offsets[si][0] if si < len(offsets) else 0
                ce = offsets[ei][1] if ei < len(offsets) else 0
                if cs < ce:
                    info['name'] = text[cs:ce].strip()
            result[et] = info
        return result


class entityextractdataset(torch.utils.data.Dataset):
    """六大实体抽取数据集：每条样本 (text, spans, presences)。

    spans:     dict[etype] -> [start_token, end_token] 或 None
    presences: dict[etype] -> 0/1
    """

    def __init__(self, texts, spans, presences):
        self.texts = texts
        self.spans = spans
        self.presences = presences

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.spans[idx], self.presences[idx]


def _entity_collate(batch):
    """自定义 collate：返回 (texts, spans, presences) 三个 list，
    避免默认 collate 对 str/dict/None 的折叠出错。"""
    return [b[0] for b in batch], [b[1] for b in batch], [b[2] for b in batch]


class entityextractor:
    """六大领域实体抽取的封装（对应 classifierfunction 的角色）。

    用法：
      ex = entityextractor(MODEL_PATH)
      ex.load_dataset('dataset.json')            # 载入标注数据
      ex.train(epochs, batch, lr, 'entity.pth')   # 训练并保存
      info = ex.extract(text, 'entity.pth')       # 抽取单条文本，返回实体信息
      ents = ex.extract_entities(text, 'entity.pth')  # 只返回 present 的实体列表
    """

    def __init__(self, model_path):
        self.model = domainentitymodel(model_path)
        self.dataset = None

    def load_dataset(self, file_path, text_key="text", entity_key="entity"):
        """载入标注数据。

        数据文件 JSON 格式：
          {"data": [{"text": "...",
                     "entity": {"product_model": [start, end] | null, ...}}]}
        entity 中每个实体类型给出字符级 [起始, 结束]（左闭右闭，实体名 = text[s:e+1]），
        null 表示该类型未出现。此处按 tokenizer 的 offset_mapping 把字符区间
        映射为 token 区间，供训练使用。
        """
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f).get("data", [])
        texts, spans, presences = [], [], []
        for item in data:
            text = item[text_key]
            ents = item.get(entity_key, {})
            enc = self.model.tokenizer(
                text, return_tensors='pt', return_offsets_mapping=True,
                truncation=True, max_length=self.model.tokenizer.model_max_length)
            offsets = enc['offset_mapping'][0].tolist()  # [(c0, c1), ...]
            span_map, pres = {}, {}
            for et in SIX_ENTITY_TYPES:
                val = ents.get(et)
                if not (isinstance(val, (list, tuple)) and len(val) >= 2):
                    span_map[et] = None
                    pres[et] = 0
                    continue
                cs, ce = val[0], val[1]
                ts = te = None
                for ti, (c0, c1) in enumerate(offsets):
                    if c0 == c1:          # 特殊 token（[CLS]/[SEP]/[PAD]）
                        continue
                    if cs < c1 and ce >= c0:   # token 与字符区间 [cs, ce] 重叠
                        if ts is None:
                            ts = ti
                        te = ti
                if ts is None:
                    span_map[et] = None
                    pres[et] = 0
                else:
                    span_map[et] = [ts, te]
                    pres[et] = 1
            texts.append(text)
            spans.append(span_map)
            presences.append(pres)
        self.dataset = entityextractdataset(texts, spans, presences)

    def train(self, epochs, batch_size, learning_rate, param_path, pre_trained_params=None):
        """训练六大实体抽取模型（冻结 BERT，只训各实体头）。"""
        if self.dataset is None:
            print("请先调用 load_dataset 载入标注数据")
            return
        if pre_trained_params is not None:
            self.model.load_state_dict(torch.load(pre_trained_params))
        self.model.train()
        self.model.bert.requires_grad_(False)
        self.model.bert.eval()
        loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, shuffle=True, collate_fn=_entity_collate)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=learning_rate, weight_decay=1e-4)
        bce = torch.nn.BCEWithLogitsLoss()
        for ep in range(epochs):
            for bid, (texts, spans, presences) in enumerate(loader):
                optimizer.zero_grad()
                presence, start, end, input_ids, _ = self.model(texts)
                B = input_ids.size(0)
                loss = torch.zeros((), device=self.model.device)
                for et in SIX_ENTITY_TYPES:
                    p_target = torch.tensor(
                        [presences[i][et] for i in range(B)],
                        dtype=torch.float32, device=self.model.device)
                    loss = loss + bce(presence[et], p_target)
                    valid = p_target > 0.5
                    if valid.any():
                        s_t = torch.tensor(
                            [spans[i][et][0] if spans[i][et] else 0 for i in range(B)],
                            dtype=torch.long, device=self.model.device)
                        e_t = torch.tensor(
                            [spans[i][et][1] if spans[i][et] else 0 for i in range(B)],
                            dtype=torch.long, device=self.model.device)
                        s_loss = torch.nn.functional.cross_entropy(
                            start[et], s_t, reduction='none')
                        e_loss = torch.nn.functional.cross_entropy(
                            end[et], e_t, reduction='none')
                        loss = loss + ((s_loss + e_loss) * valid).sum() / valid.sum().clamp(min=1)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                if bid % 10 == 0:
                    print("轮次:%d\t样本数:%d\t损失:%.5f" % (ep + 1, bid + 1, loss.item()))
        print("正在保存六大实体抽取模型参数")
        torch.save(self.model.state_dict(), param_path)

    def extract(self, text, param_path=None, presence_threshold=0.5):
        """抽取单条文本的六大实体，返回 dict[etype] -> 实体信息。"""
        if param_path is not None:
            self.model.load_state_dict(
                torch.load(param_path, map_location=self.model.device))
        return self.model.extract(text, presence_threshold)

    def extract_entities(self, text, param_path=None, presence_threshold=0.5):
        """抽取并只返回 present 的实体列表（每个元素含 etype/name/confidence/attributes）。"""
        info = self.extract(text, param_path, presence_threshold)
        out = []
        for et in SIX_ENTITY_TYPES:
            if info[et]['present']:
                out.append(dict(info[et], etype=et))
        return out
if __name__=="__main__":
    print("Welcome To Marisa Forge")
    print("Instructions:")
    print("1.“classifiermodel”是一个分类模型，你需要设定模型文件、输出维数和隐藏层。\n模型文件用于bert处理。")
    print("2.“classifierfunction”是分类函数，用于加载数据集和进行分类。")
    print("3.load_dataset的用法")
    print("\t3.1数据集以json格式呈现，请使用{\"data:\"[]}存储，内部各元素建议包含text与label字段。如果有其他字段，请在data_keywords和labels_keywords中指定")
    print("\t3.2请确保label是以数组形式存储的，且label长度应与输出维数一致。如果不一致，会视情况选择截断或使用0补齐")
    print("\t3.3如果使用了single_domain，请额外在数据集文件内添加一个is_negative字段，并对负样本标记为true")
    print("4.single_domain的说明：single_domain是针对垂直领域分类的，垂直领域分类的高维空间分布和水平领域分类不同。\n后者为壳表面分布，前者是超立方体内部均匀分布。这将导致余弦相似度失效。\n因而针对垂直领域，数据集的is_negative会用于过滤器训练。被过滤器过滤的文本统一返回零向量以保证输出值为0")
    print("5.训练分为两步：train_classifier(...) 训练主题分类器，train_filter(...) 训练过滤器；也可用 train(...) 一次性训练两者")
    print("6.分类函数直接加载参数，不再另行训练")
    s=input("输入Y查看演示...")
    if(s=='Y' or s=='y'):
        MODEL_PATH=os.path.dirname(__file__)+"/models/paraphrase-multilingual-MiniLM-L12-v2"
        print("使用\"classifierfunction(MODEL_PATH,3,[100,10],True)\"初始化")
        cEDU=classifierfunction(MODEL_PATH,3,[100,10],True)
        print("使用\"load_dataset(os.path.dirname(__file__)+\"/merged_dataset.json\")\"加载数据集")
        cEDU.load_dataset(os.path.dirname(__file__)+"/merged_dataset.json")
        print("是否开始训练? Y/N")
        if(input() in ("Y","y")):
            print("使用train函数训练模型，参数为(数据集,轮次,批量大小,学习率,参数保存路径,预训练参数路径（默认为空）)")
            print("加载预训练参数以从上一次训练结果继续调整,如果留空则重新开始训练")
            cEDU.train(cEDU.dataset,30,16,0.001,os.path.dirname(__file__)+"/params.pth")#轮次至少30，10轮欠训练，损失会卡在0.63附近
        text=input("输入文本:")
        cEDU.test(text,os.path.dirname(__file__)+"/params.pth",os.path.dirname(__file__)+"/filterparam.pth")