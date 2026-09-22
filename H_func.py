# -*-coding: utf-8 -*-
import transformers
import torch
from transformers import BertTokenizer, BertModel
import math
import jieba
import jieba.analyse as analyse
import os
import sys
import sentence_transformers
#学习原型机
'''
H=entropy(text)*cosine_similarity(domain_weight_vector,domain_weight_req)
其中：两个领域是通过监督学习预测得到的向量进行零中心化后得到的标准向量
这里假设各样本是分布在超立方体壳上的（超立方体壳厚度小，可以用角度衡量）
针对垂直领域，需要考虑壳厚度
'''
import torch.nn as nn


# 输出示例: [0.12, 0.03, 0.89, 0.01, 0.45, ...]
# 表示该文本在"科技"领域权重0.89，在"财经"领域权重0.45
###领域标签设计为控件，以便用户自主配置
#######-----------Starting-----------------####
#识文解意的爱书人明白文字中的价值。
DFSTOPWORDPATH=os.path.dirname(os.path.abspath(__file__))+'/stopwords.txt'
SIM_THRESHOLD=0.6
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_DIR, 'models', 'paraphrase-multilingual-MiniLM-L12-v2')
WEIGHTS_PATH = os.path.join(PROJECT_DIR, 'model_params.pth')
# THUCnews 数据集目录（默认位置在项目上级目录的训练集里，可自行修改）
DATASET_PATH = os.path.normpath(os.path.join(PROJECT_DIR, '..', '..', '训练集', 'Dataset', 'THUCNews', 'THUCNews'))

DOMAINS = ["彩票", "体育", "星座", "房产", "家居", "教育", "科技", "时尚", "时政", "游戏", "娱乐", "社会", "股票"]

# 请求句模板：用领域名填充，构造「用户请求式」短样本
REQUEST_TEMPLATES = [
    "我想了解{domain}方面的信息",
    "给我讲讲{domain}的最新动态",
    "关于{domain}的内容有哪些",
    "有没有{domain}相关的资料",
    "请介绍一下{domain}",
    "我需要{domain}方面的帮助",
    "{domain}最近有什么新进展",
]

# 各领域的常见关键词，用于强化短请求文本的判别
DOMAIN_KEYWORDS = {
    "彩票": ["彩票", "福彩", "体彩", "中奖", "开奖", "双色球", "大乐透"],
    "体育": ["足球", "篮球", "比赛", "球员", "联赛", "奥运", "冠军"],
    "星座": ["星座", "白羊座", "金牛座", "双子座", "运势", "占星", "星盘"],
    "房产": ["房产", "房价", "楼盘", "买房", "二手房", "开发商", "楼市"],
    "家居": ["家居", "装修", "家具", "家装", "建材", "卧室", "卫浴"],
    "教育": ["教育", "学校", "高考", "大学", "课程", "教师", "学生"],
    "科技": ["科技", "人工智能", "互联网", "手机", "芯片", "软件", "机器人"],
    "时尚": ["时尚", "服装", "潮流", "穿搭", "品牌", "时装", "模特"],
    "时政": ["时政", "政策", "政府", "外交", "会议", "改革", "国际"],
    "游戏": ["游戏", "电竞", "手游", "网游", "玩家", "版本", "赛事"],
    "娱乐": ["娱乐", "明星", "电影", "电视剧", "综艺", "演员", "音乐"],
    "社会": ["社会", "民生", "新闻", "事件", "警方", "市民", "事故"],
    "股票": ["股票", "股市", "股民", "大盘", "证券", "涨停", "基金"],
}


def get_bigram_tf(word):
        # 得到二元词的词频表
        bigram_tf = {}
        for i in range(len(word) - 1):
            bigram_tf[(word[i], word[i + 1])] = bigram_tf.get(
                (word[i], word[i + 1]), 0) + 1
        return bigram_tf
def entropy(text,stopword=DFSTOPWORDPATH):
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
def keyword_match(text:str, req:str, model_path):
    kwtext = analyse.extract_tags(text, withWeight=True, allowPOS=('ns', 'n', 'vn', 'v'))
    kwreq  = analyse.extract_tags(req,  withWeight=True, allowPOS=('ns', 'n', 'vn', 'v'))
    Tseg = jieba.lcut(text, use_paddle=True)
    p = len(kwtext)
    q = len(kwreq)
    if p == 0 or q == 0:            # 任一侧抽不到关键词
        return 0.0

    model = sentence_transformers.SentenceTransformer(model_path)

    # 预编码关键词，避免在内层循环反复 encode（原来每个 (i,j) 都 encode 一次）
    emb_text = [model.encode(kwtext[i][0]) for i in range(p)]
    emb_req  = [model.encode(kwreq[j][0])  for j in range(q)]

    # 相似度矩阵：model.similarity 返回 (1,1) 张量，这里取出标量 float
    sim = [[0.0] * q for _ in range(p)]
    for i in range(p):
        for j in range(q):
            sim[i][j] = float(model.similarity(emb_text[i], emb_req[j]).item())

    # dp 尺寸 (p+1) x (q+1)，第 0 行 / 第 0 列全为 0 作为边界。
    # 原来直接用 dp[i-1][j]、dp[i-1][j-1]、dp[i][j-1]，当 i=0 或 j=0 时会取到
    # 负下标（dp[-1] = 最后一行、dp[i][-1] = 最后一列），把矩阵首尾相接成环，
    # 导致同一权重被重复累加、每行都收敛成同一个值，match_score 被算到 >1。
    dp   = [[0.0] * (q + 1) for _ in range(p + 1)]
    step = [[None] * (q + 1) for _ in range(p + 1)]
    for i in range(1, p + 1):
        for j in range(1, q + 1):
            s1 = dp[i - 1][j]                       # 跳过 text 关键词 i-1
            s2 = dp[i - 1][j - 1]                   # 匹配 text[i-1] 与 req[j-1]
            if sim[i - 1][j - 1] > SIM_THRESHOLD:
                s2 += kwreq[j - 1][1] * sim[i - 1][j - 1]
            s3 = dp[i][j - 1]                       # 跳过 req 关键词 j-1
            best = s1
            step[i][j] = (i - 1, j)
            if s2 > best:
                best = s2
                step[i][j] = (i - 1, j - 1)
            if s3 > best:
                best = s3
                step[i][j] = (i, j - 1)
            dp[i][j] = best

    total_req = sum([x[1] for x in kwreq])
    match_score = dp[p][q] / total_req if total_req > 0 else 0.0

    # 回溯匹配路径，找出真正被匹配上的 text 关键词
    path = []
    i, j = p, q
    while i > 0 and j > 0:
        pi, pj = step[i][j]
        if (pi, pj) == (i - 1, j - 1):              # 对角线 => text[i-1] 被匹配
            path.append(i - 1)
        i, j = pi, pj
    matched = [kwtext[i][0] for i in set(path)]

    # 定位匹配关键词在原文分词中的位置，用于计算密集度
    locations = []
    for w in matched:
        s = -1
        try:
            s = Tseg.index(w)
        except ValueError:
            # 关键词不在分词结果中时，用 embedding 相似度找最近的分词位置
            v1 = model.encode(w)
            best_sim = -1.0
            for idx, seg in enumerate(Tseg):
                sim_val = float(model.similarity(v1, model.encode(seg)).item())
                if sim_val > best_sim:
                    best_sim = sim_val
                    s = idx
            if best_sim <= SIM_THRESHOLD:
                s = -1
        if s != -1:
            locations.append(s)

    if len(locations) == 0:
        return 0.0

    span = (max(locations) - min(locations) + 1) / len(Tseg)
    density_score = math.exp(-2 * span)
    return match_score * density_score
    ###带权的最长子序列
    #关键词应当以名词为主，动词为次进行
    #关键词密集出现比分散出现的信息更容易强调客户端要求，且根据人的注意力衰减机制，如果文字太长关键词过于分散会导致人的兴趣下降、阅读困难
    #对于返回值为0的情况：如果返回值为0，说明存在两种情况：一是用户表述笼统或者只需要领域，二是不符合具体要求事项，此时H非负时应该取最小值而不是0
#向量分类函数作为可选项，允许开发者自行设计，默认使用torch.nn的监督学习
def H_func(text:str,req:str,classifier_function,kw_model=MODEL_PATH):
    entr=entropy(text)
    domain_weight_vector=classifier_function(text)
    domain_weight_req=classifier_function(req)
    v = domain_weight_vector - domain_weight_vector.mean()
    r = domain_weight_req - domain_weight_req.mean()
    dot_product = float((v * r).sum())
    norm_a = math.sqrt(float((v * v).sum()))
    norm_b = math.sqrt(float((r * r).sum()))
    if norm_a == 0 or norm_b == 0:
        cosine_similarity = 0
    else:
        cosine_similarity = dot_product / (norm_a * norm_b)
    match_rank=keyword_match(text,req,kw_model)
    return [entr * (abs(cosine_similarity)**(2-match_rank)),cosine_similarity,match_rank,domain_weight_vector,domain_weight_req]
class domain_classifier(torch.nn.Module):
    def __init__(self, num_domains,model=MODEL_PATH):
        super().__init__()
        self.bert = BertModel.from_pretrained(model)
        hidden_size = self.bert.config.hidden_size  # 从模型配置动态获取 hidden_size（MiniLM=384, BERT-base=768）
        self.middle_layer = torch.nn.Linear(hidden_size, 233)
        self.classifier=torch.nn.Linear(233,num_domains)
    def forward(self, input_ids, attention_mask,using_pooler=True):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        # pooler_output 可能为 None（某些 SentenceTransformer 模型没有 pooler），
        # 此时回退到对 last_hidden_state 做 mean pooling。
        # 注意：混合训练会把短文本和长文本放进同一个 batch，若不屏蔽 padding，
        # 短文本会被大量 [PAD] 向量稀释，重新塌缩成「平行于全1向量」的平坦输出。
        if using_pooler==True:
            cls_vec = outputs.pooler_output
            if cls_vec is None:
                mask = attention_mask.unsqueeze(-1).to(outputs.last_hidden_state.dtype)
                cls_vec = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            cls_vec=outputs.last_hidden_state[:, 0]
        logits = self.middle_layer(cls_vec)
        Lrelu = torch.nn.LeakyReLU(negative_slope=1e-2)  
        middleweights=Lrelu(logits)
        output_logits=self.classifier(middleweights)
        weights=torch.sigmoid(output_logits)# 多标签输出：每个领域独立 0~1，不强制和为1
        return weights


def _read_thucnews(file_path):
    """读取一篇 THUCnews 新闻，返回 (标题, 正文)。第一行是标题，其余是正文。"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
    title = lines[0] if lines else ""
    body = "\n".join(lines[1:])
    return title, body


def build_mixed_dataset(dataset_path=DATASET_PATH, domains=DOMAINS,
                        samples_per_domain=100, short_len=80):
    """构造混合训练集：标题 + 短文本(正文截断) + 长文字(全文) + 请求句 + 领域词/关键词。

    标签仍是 one-hot（单标签），输出保持 sigmoid 多标签不变。
    """
    texts, labels = [], []
    n = len(domains)
    for i, domain in enumerate(domains):
        one_hot = [0] * n
        one_hot[i] = 1
        domain_path = os.path.join(dataset_path, domain)
        for fn in sorted(os.listdir(domain_path))[:samples_per_domain]:
            title, body = _read_thucnews(os.path.join(domain_path, fn))
            # 长文字：标题 + 正文（与原训练一致，训练分布内的长文本）
            long_text = (title + "\n" + body).strip()
            if long_text:
                texts.append(long_text)
                labels.append(list(one_hot))
            # 标题：本身就是一条短文本（新闻标题）
            if title:
                texts.append(title)
                labels.append(list(one_hot))
            # 短文本：正文前 short_len 字，模拟短请求/摘要
            if len(body) >= short_len:
                texts.append(body[:short_len])
                labels.append(list(one_hot))
        # 请求句：模板 + 领域名
        for tpl in REQUEST_TEMPLATES:
            texts.append(tpl.format(domain=domain))
            labels.append(list(one_hot))
        # 领域名本身 + 关键词：直接教会分类头「短词 → 对应领域尖峰」
        texts.append(domain)
        labels.append(list(one_hot))
        for kw in DOMAIN_KEYWORDS.get(domain, []):
            texts.append(kw)
            labels.append(list(one_hot))
    return texts, labels

class TextDataset(torch.utils.data.Dataset):
        def __init__(self, texts, labels):
            self.texts = texts
            self.labels = labels
        def __len__(self):
            return len(self.texts)
        def __getitem__(self, idx):
            return self.texts[idx], self.labels[idx]
def train_domain_classifier(dataset_path=DATASET_PATH, domains=DOMAINS,
                            samples_per_domain=100, short_len=80,
                            epochs=10, batch_size=32, lr=2e-5, freeze_bert=False,
                            save_path=WEIGHTS_PATH):
    """混合数据 + sigmoid 多标签（BCELoss）训练。

    默认微调整个 MiniLM（AdamW lr=2e-5）：比冻结编码器的线性探针更强，
    能给出更准确、更尖锐的领域权重。若只想快速只训分类头，设 freeze_bert=True
    并把 lr 提到 1e-3（但准确率/尖锐度会下降，短文本易出现错峰）。

    运行方式： python H_func.py --train
    """
    

    texts, labels = build_mixed_dataset(dataset_path, domains, samples_per_domain, short_len)
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    model = domain_classifier(num_domains=len(domains))
    if freeze_bert:
        model.bert.requires_grad_(False)  # 冻结编码器，只训练分类头
        model.bert.eval()                 # 冻结层关闭 dropout，输出更稳定
    dset = TextDataset(texts, labels)
    loader = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=True)
    loss_fn = torch.nn.BCELoss()  # 多标签分类损失（接收 sigmoid 输出，不需要内部再做 sigmoid）
    # 只优化 requires_grad=True 的参数（冻结 BERT 时即只剩分类头）
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=lr, weight_decay=1e-2)

    for i in range(epochs):
        for bid, batch in enumerate(loader):
            batch_texts, batch_labels = batch
            inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True)
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask,using_pooler=False)  # forward 返回 sigmoid 权重
            # DataLoader 可能返回 list of tensors 或单个 tensor，统一处理
            if isinstance(batch_labels, torch.Tensor):
                labels_tensor = batch_labels.float()
            else:
                labels_tensor = torch.stack(batch_labels).float()
            # 确保 labels 形状与 outputs 一致 [batch_size, num_domains]
            if labels_tensor.shape != outputs.shape:
                labels_tensor = labels_tensor.T
            loss = loss_fn(outputs, labels_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 防梯度爆炸
            optimizer.step()

            if bid % 10 == 0:
                print(f"Epoch {i+1}, Batch {bid+1}, Loss: {loss.item()}")
    # 训练后自检：打印探针文本的领域权重，确认输出不再平坦
    model.eval()
    probe_texts = ["体育", "彩票行情", "我需要了解人工智能技术的新应用",
                   "新型摄像头+人工智能技术将用于足球裁判，有望提高判罚的准确性。"]
    DEFAULT_CASES = [
    ("人工智能芯片研发取得新突破", "科技"),
    ("国产大模型正式发布参数破千亿", "科技"),
    ("量子计算原型机实现里程碑突破", "科技"),
    ("新款智能手机发布会定档下周", "科技"),
    ("国足昨晚比赛获胜晋级下一轮", "体育"),
    ("中国男篮夺得亚洲杯冠军", "体育"),
    ("某明星官宣恋情引发热议", "娱乐"),
    ("暑期档电影票房突破50亿", "娱乐"),
    ("今日大盘上涨券商板块领涨", "股票"),
    ("股市午后跳水创业板指跌超2%", "股票"),
    ("央行宣布下调存款准备金率", "时政"),
    ("国务院出台稳经济一揽子政策", "时政"),
    ("教育部发布高等教育改革方案", "教育"),
    ("高考成绩今日公布考生可查分", "教育"),
    ("台风来袭沿海多城市停课停运", "社会"),
    ("某地发生燃气泄漏事故紧急处置", "社会"),
    ("某地楼市新政首付比例下调", "房产"),
    ("某热门手游新版本今日上线", "游戏"),
    ("双色球今晚开奖头奖井喷", "彩票"),
]
    probe_texts+=[x[0] for x in DEFAULT_CASES]
    print("训练后自检（Top3 领域权重）：")
    with torch.no_grad():
        for pt in probe_texts:
            inputs = tokenizer(pt, return_tensors="pt", truncation=True, padding=True)
            w = model(inputs['input_ids'], inputs['attention_mask'],using_pooler=False).squeeze(0).numpy()
            top = sorted(zip(domains, w), key=lambda x: -x[1])[:3]
            print(f"  「{pt}」-> " + " ".join(f"{d}:{v:.3f}" for d, v in top))

    torch.save(model.state_dict(), save_path)
    print(f"模型参数已保存到 {save_path}")


if __name__ == "__main__":
    if "--train" in sys.argv:
        if len(sys.argv)>1:
            DATASET_PATH=sys.argv[1]
        train_domain_classifier()
        sys.exit(0)
    # 只加载一次模型和 tokenizer（旧版每次调用 classifier 都重建模型并读 470MB 权重）
    model = domain_classifier(num_domains=len(DOMAINS))
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=torch.device('cpu')))
    model.eval()
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)

    def classifier(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            # 必须与训练时的 using_pooler=False 保持一致（用 CLS token），否则训练/推理特征不一致，
            # 分类头会输出接近平坦的 ~0.5，测试 loss 会卡在 ln2≈0.69 附近。
            domain_weight_vector = model(inputs['input_ids'], inputs['attention_mask'], using_pooler=False).squeeze(0).numpy()  # 长度为K的ndarray
        return domain_weight_vector
    DEFAULT_CASES = [
    ("人工智能芯片研发取得新突破", "科技"),
    ("国产大模型正式发布参数破千亿", "科技"),
    ("量子计算原型机实现里程碑突破", "科技"),
    ("新款智能手机发布会定档下周", "科技"),
    ("国足昨晚比赛获胜晋级下一轮", "体育"),
    ("中国男篮夺得亚洲杯冠军", "体育"),
    ("某明星官宣恋情引发热议", "娱乐"),
    ("暑期档电影票房突破50亿", "娱乐"),
    ("今日大盘上涨券商板块领涨", "股票"),
    ("股市午后跳水创业板指跌超2%", "股票"),
    ("央行宣布下调存款准备金率", "时政"),
    ("国务院出台稳经济一揽子政策", "时政"),
    ("教育部发布高等教育改革方案", "教育"),
    ("高考成绩今日公布考生可查分", "教育"),
    ("台风来袭沿海多城市停课停运", "社会"),
    ("某地发生燃气泄漏事故紧急处置", "社会"),
    ("某地楼市新政首付比例下调", "房产"),
    ("某热门手游新版本今日上线", "游戏"),
    ("双色球今晚开奖头奖井喷", "彩票"),
]
    dataset=[]
    req="高考最新动态"
    for i in DEFAULT_CASES:
        s=[0]*13
        s[DOMAINS.index(i[1])]=1
        dataset.append((i[0],s))
    dataset.append((req,[0,0,0,0,0,1,0,0,0,0,0,0,0]))
    texts=[]
    labels=[]
    for i in dataset:
        texts.append(i[0])
        labels.append(i[1])
    dset=TextDataset(texts,labels)
    print("请求：",req)
    print("分类结果:",classifier(req))
    loss=torch.nn.BCELoss()
    total_loss=0.0
    p=0
    for text,label in dset:  # 直接按样本迭代，避免 DataLoader 打包后再解包错位
        result=H_func(text,req,classifier)
        pred=torch.tensor(result[3],dtype=torch.float32)  # result[3] 是文本自身的领域权重向量
        gt  =torch.tensor(label,dtype=torch.float32)      # label 转为浮点以匹配 BCELoss
        sample_loss=loss(pred,gt)
        total_loss+=sample_loss.item()
        p+=1
        print("新闻:",text)
        print("价值参数:",result[0:3])
        print("loss:",sample_loss.item())
    print("平均 loss:",total_loss/p if p else 0.0)
    #短文本Loss还是高