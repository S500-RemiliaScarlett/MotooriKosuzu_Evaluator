# -*- coding: utf-8 -*-
"""
知识图谱管理工具（Knowledge Graph Management Tool）
==================================================

在 KNgraph.py 的基础图谱结构（entity / relation / KnowledgeGraph / evententity）
之上，依据《Day2交付物.md》的「价值判定函数」实现一套完整的知识图谱管理与分析工具。

与 KNgraph.py 的对应关系
------------------------
  KNgraph.py 的 entity      -> 本文件 Entity（增强：哈希/相等/序列化）
  KNgraph.py 的 relation    -> 本文件 Relation（增强：方向属性、支撑权重）
  KNgraph.py 的 evententity -> 本文件 EventEntity（增强：时间/观点方向字段）
  KNgraph.py 的 KnowledgeGraph -> 本文件 KnowledgeGraph（增强：邻接、最短路、持久化、分析）
  KNgraph.py 的 TPlist      -> 本文件 RELATION_TYPES

功能总览
--------
1. 图谱管理
   - 实体 / 关系 / 三元组的增删改查
   - 邻接查询、最短路（支持有向 / 无向、按关系类型过滤）
   - 序列化（JSON 保存 / 加载）
2. 冗余度 Sim 计算
   - Hash 层：SimHash 计算哈希相似度 S，排除 S > alpha 的重复文本
   - 一义多表述修正层：sim(x,y) = S(x,y) * exp(-D(x,y) + 1)，D 为相似图最短路
3. 相似事件群
   - 相似图 G 构建（sim > 0.8 连边）
   - 谱聚类分割（numpy 可用时；否则回退连通分量）
   - 每群取 H 最大者为中心
4. 传播影响力
   - tanh(N + log2(M) + min(1, log10(X/T)))
5. 关联支撑度
   - 混合有向/无向图统一为有向图，求入边支撑贡献之和
   - 额外证据补正：因果图根节点最短路 1/D
6. 多源交叉印证 Q 与归一化支撑度 S_r = sigmoid(Q + S)
7. 价值判定 H = M * DS(T) * (1 - Sim(T))

说明：本工具聚焦图谱侧计算；FMatch / KWMatch 等依赖领域分类器与关键词模型的
部分作为外部输入（见 value_H 参数说明），与 H_func.py 配合使用。
"""

import json
import math
import os
import hashlib
from collections import defaultdict, deque

# ---------------------------------------------------------------------------
# 常量定义
# ---------------------------------------------------------------------------

# 关系类型：与 KNgraph.py 的 TPlist 保持一致
#   reason     因果边（有向）
#   time       时间先后边（有向）
#   similarity 相似边（无向）
#   topicsim   共主题边（无向，按观点方向区分一致/相反）
RELATION_TYPES = ['reason', 'time', 'similarity', 'topicsim']

# 关联支撑度中各类型边的支撑贡献（《Day2交付物.md》）
#   因果边为 1，相似为 0.8，共主题、观点方向一致为 0.6，共主题、观点相反为 -1
#   time 边在 Day2 中未单独给出权重，此处默认不计入支撑贡献（仅用于时序排序），
#   如需赋值可修改 SUPPORT_WEIGHT['time']。
SUPPORT_WEIGHT = {
    'reason': 1.0,      # 因果边
    'time': 0.0,        # 时间先后（Day2 未明确，默认不贡献支撑）
    'similarity': 0.8,  # 相似边
    'topicsim': 0.6,    # 共主题、观点方向一致（相反时为 -1，见 _support_contribution）
}

# 相似图连边阈值 / 冗余去重阈值
SIM_EDGE_THRESHOLD = 0.8   # sim > 0.8 连边
DUP_ALPHA = 0.95           # SimHash 相似度 S > alpha 视为重复文本


# ---------------------------------------------------------------------------
# 基础类（增强版 KNgraph.py 结构）
# ---------------------------------------------------------------------------

class Entity:
    """实体父类（对应 KNgraph.py 的 entity）。"""

    def __init__(self, Id, text):
        self.Id = Id
        self.text = text

    def __hash__(self):
        return hash(self.Id)

    def __eq__(self, other):
        return isinstance(other, Entity) and self.Id == other.Id

    def __repr__(self):
        return f"Entity(Id={self.Id})"

    def to_dict(self):
        return {"Id": self.Id, "text": self.text}

    @classmethod
    def from_dict(cls, d):
        return cls(d["Id"], d["text"])


class Relation:
    """关系父类（对应 KNgraph.py 的 relation）。

    direction 字段用于共主题边（topicsim）标记观点方向：
        'same'    —— 观点方向大致一致（支撑贡献 0.6）
        'opposite'—— 观点方向相反（支撑贡献 -1）
    其余类型可忽略该字段。
    """

    def __init__(self, Id1, Id2, tp, direction=None):
        self.Id1 = Id1
        self.Id2 = Id2
        self.tp = tp
        self.direction = direction

    def __hash__(self):
        return hash((self.Id1, self.Id2, self.tp, self.direction))

    def __eq__(self, other):
        return (isinstance(other, Relation) and
                (self.Id1, self.Id2, self.tp, self.direction) ==
                (other.Id1, other.Id2, other.tp, other.direction))

    def __repr__(self):
        return f"Relation({self.Id1} -{self.tp}-> {self.Id2})"

    def to_dict(self):
        return {"Id1": self.Id1, "Id2": self.Id2,
                "tp": self.tp, "direction": self.direction}

    @classmethod
    def from_dict(cls, d):
        return cls(d["Id1"], d["Id2"], d["tp"], d.get("direction"))


class EventEntity(Entity):
    """事件实体（对应 KNgraph.py 的 evententity）。"""

    def __init__(self, Id, text, platform=None, reactiondata=None, topic=None,
                 time=None, view=None):
        super().__init__(Id, text)
        self.platform = platform          # 来源平台（信源）
        self.reactiondata = reactiondata or {}  # 互动数据表，如 {'comments':..,'likes':..}
        self.topic = topic                # 主题标签
        self.time = time                  # 事件时间（用于时间跨度的计算）
        self.view = view                  # 观点方向（'same'/'opposite'，可为 None）

    def to_dict(self):
        d = super().to_dict()
        d.update({"platform": self.platform, "reactiondata": self.reactiondata,
                  "topic": self.topic, "time": _to_serializable(self.time),
                  "view": self.view})
        return d

    @classmethod
    def from_dict(cls, d):
        e = cls(d["Id"], d["text"], d.get("platform"), d.get("reactiondata"),
                d.get("topic"), _parse_time(d.get("time")), d.get("view"))
        return e


# ---------------------------------------------------------------------------
# SimHash（Hash 层）
# ---------------------------------------------------------------------------

class SimHash:
    """SimHash 算法实现（用于冗余度 Sim 的 Hash 层）。

    对文本抽取字符 n-gram 特征，每个特征哈希到 bits 位并加权累加后二值化；
    两条文本的相似度 = 1 - 汉明距离 / bits。
    字符 n-gram 对中文文本无需分词即可工作，保持本工具零额外依赖。
    """

    def __init__(self, bits=64, n_gram=3):
        self.bits = bits
        self.n_gram = n_gram

    def _features(self, text):
        t = ''.join(text.split())  # 去空白
        n = self.n_gram
        if len(t) < n:
            return [t] if t else []
        return [t[i:i + n] for i in range(len(t) - n + 1)]

    @staticmethod
    def _hash(token):
        # 取 md5 前 8 字节作为 64 位哈希
        return int.from_bytes(hashlib.md5(token.encode('utf-8')).digest()[:8], 'little')

    def fingerprint(self, text):
        v = [0] * self.bits
        for feat in self._features(text):
            h = self._hash(feat)
            for i in range(self.bits):
                v[i] += 1 if (h >> i) & 1 else -1
        fp = 0
        for i in range(self.bits):
            if v[i] >= 0:
                fp |= (1 << i)
        return fp

    def similarity(self, text1, text2):
        """返回 [0,1] 的哈希相似度 S。"""
        fp1 = self.fingerprint(text1)
        fp2 = self.fingerprint(text2)
        dist = bin(fp1 ^ fp2).count('1')
        return 1.0 - dist / self.bits


# ---------------------------------------------------------------------------
# 图谱管理（增强版 KnowledgeGraph）
# ---------------------------------------------------------------------------

class KnowledgeGraph:
    """知识图谱：管理实体 / 关系 / 三元组，并实现 Day2 的分析函数。"""

    def __init__(self, simhash=None):
        self.entities = {}           # Id -> Entity（以字典管理，便于按 Id 检索）
        self.relations = set()       # Relation 集合
        self.triples = []            # (head_Id, relation, tail_Id)
        self.simhash = simhash or SimHash()

    # ------------------------------------------------------------------
    # 基础管理（增删改查）
    # ------------------------------------------------------------------
    def add_entity(self, entity):
        """添加实体。"""
        self.entities[entity.Id] = entity

    def get_entity(self, Id):
        """按 Id 获取实体，不存在返回 None。"""
        return self.entities.get(Id)

    def remove_entity(self, Id):
        """删除实体，并级联删除与之相关的所有关系与三元组。"""
        if Id not in self.entities:
            return False
        del self.entities[Id]
        self.relations = {r for r in self.relations
                          if r.Id1 != Id and r.Id2 != Id}
        self.triples = [t for t in self.triples
                        if t[0] != Id and t[2] != Id]
        return True

    def add_relation(self, relation):
        """添加关系（边）。"""
        self.relations.add(relation)

    def remove_relation(self, relation):
        """删除关系。"""
        self.relations.discard(relation)
        self.triples = [t for t in self.triples if t[1] != relation]

    def add_triple(self, head_Id, relation, tail_Id):
        """添加三元组（head_Id / tail_Id 需为已存在实体 Id）。"""
        if head_Id in self.entities and tail_Id in self.entities and relation in self.relations:
            self.triples.append((head_Id, relation, tail_Id))
        else:
            raise ValueError("实体或关系不存在")

    def get_entities(self):
        return list(self.entities.values())

    def get_relations(self, tp=None):
        if tp is None:
            return list(self.relations)
        return [r for r in self.relations if r.tp == tp]

    def get_triples(self):
        return self.triples

    def __len__(self):
        return len(self.entities)

    # ------------------------------------------------------------------
    # 邻接与最短路
    # ------------------------------------------------------------------
    def _similarity_adj(self):
        """相似图邻接表（无向，仅 similarity 边）。"""
        adj = defaultdict(set)
        for r in self.relations:
            if r.tp == 'similarity':
                adj[r.Id1].add(r.Id2)
                adj[r.Id2].add(r.Id1)
        return adj

    def _directed_edges(self):
        """统一处理后的有向边列表 [(src, dst, contribution)]。

        有向边（reason / time）按 Id1 -> Id2；无向边（similarity / topicsim）
        视作方向相反的两条平行有向边。
        """
        edges = []
        for r in self.relations:
            w = self._support_contribution(r)
            if r.tp in ('reason', 'time'):
                edges.append((r.Id1, r.Id2, w))
            else:
                edges.append((r.Id1, r.Id2, w))
                edges.append((r.Id2, r.Id1, w))
        return edges

    def _causal_adj(self):
        """因果图邻接表（有向，仅 reason 边）。"""
        adj = defaultdict(set)
        indeg = defaultdict(int)
        for r in self.relations:
            if r.tp == 'reason':
                adj[r.Id1].add(r.Id2)
                indeg[r.Id2] += 1
                indeg.setdefault(r.Id1, 0)
        return adj, indeg

    @staticmethod
    def _shortest_path(adj, a, b):
        """BFS 求最短路长（边数），不可达返回 None。"""
        if a == b:
            return 0
        if a not in adj:
            return None
        queue = deque([(a, 0)])
        seen = {a}
        while queue:
            node, d = queue.popleft()
            for nb in adj.get(node, ()):
                if nb == b:
                    return d + 1
                if nb not in seen:
                    seen.add(nb)
                    queue.append((nb, d + 1))
        return None

    def shortest_path(self, a, b, tp=None):
        """两实体间最短路（默认在相似图内；tp='reason' 时在因果图内）。"""
        if tp == 'reason':
            adj, _ = self._causal_adj()
        else:
            adj = self._similarity_adj()
        return self._shortest_path(adj, a, b)

    # ------------------------------------------------------------------
    # 冗余度 Sim
    # ------------------------------------------------------------------
    def simhash_similarity(self, a, b):
        """Hash 层相似度 S（a/b 可为实体对象或 Id）。"""
        ea = a if isinstance(a, Entity) else self.entities.get(a)
        eb = b if isinstance(b, Entity) else self.entities.get(b)
        if ea is None or eb is None:
            return 0.0
        return self.simhash.similarity(ea.text, eb.text)

    def similarity(self, a, b):
        """sim(x,y) = S(x,y) * exp(-D(x,y) + 1)。

        D 为两节点在相似图中的最短路长：
          D=1 时 exp(0)=1，取哈希相似度本身；
          D 增大时先迅速衰减，D 很大时接近 0 且衰减变慢；
          不可达（D=None）时视为 0。
        """
        ea = a if isinstance(a, Entity) else self.entities.get(a)
        eb = b if isinstance(b, Entity) else self.entities.get(b)
        if ea is None or eb is None:
            return 0.0
        S = self.simhash_similarity(ea, eb)
        D = self.shortest_path(ea.Id, eb.Id)
        if D is None:
            return 0.0
        return S * math.exp(-D + 1)

    def redundancy(self, entity, candidates=None):
        """Sim(T) = 其他文本与 T 的 sim 的最大值。"""
        e = entity if isinstance(entity, Entity) else self.entities.get(entity)
        if e is None:
            return 0.0
        others = candidates if candidates is not None else self.get_entities()
        best = 0.0
        for o in others:
            if o.Id == e.Id:
                continue
            best = max(best, self.similarity(e, o))
        return best

    def deduplicate(self, alpha=DUP_ALPHA):
        """Hash 层去重：返回与某已存在文本 S>alpha 的重复实体 Id 列表（不删除）。"""
        dup = []
        ents = self.get_entities()
        for i in range(len(ents)):
            for j in range(i + 1, len(ents)):
                if self.simhash_similarity(ents[i], ents[j]) > alpha:
                    dup.append((ents[i].Id, ents[j].Id))
        return dup

    # ------------------------------------------------------------------
    # 相似事件群
    # ------------------------------------------------------------------
    def build_similarity_graph(self, threshold=SIM_EDGE_THRESHOLD):
        """构建相似图 G = (N, E)，E = {sim(x,y) > threshold}，返回邻接表。"""
        adj = defaultdict(set)
        ents = self.get_entities()
        for i in range(len(ents)):
            for j in range(i + 1, len(ents)):
                if self.simhash_similarity(ents[i], ents[j]) > threshold:
                    adj[ents[i].Id].add(ents[j].Id)
                    adj[ents[j].Id].add(ents[i].Id)
        return adj

    def similar_event_groups(self, threshold=SIM_EDGE_THRESHOLD):
        """对相似图 G 做连通分量分割，返回 [[Id, ...], ...]。

        谱聚类的兜底实现；需要更细粒度分割时调用 spectral_cluster。
        """
        adj = self.build_similarity_graph(threshold)
        seen = set()
        groups = []
        for Id in self.entities:
            if Id in seen:
                continue
            comp = []
            queue = deque([Id])
            seen.add(Id)
            while queue:
                node = queue.popleft()
                comp.append(node)
                for nb in adj.get(node, ()):
                    if nb not in seen:
                        seen.add(nb)
                        queue.append(nb)
            groups.append(comp)
        return groups

    def spectral_cluster(self, nodes, k=None):
        """对一组节点做谱聚类（需 numpy）；k=None 时用特征值间隙启发式选 k。"""
        try:
            import numpy as np
        except ImportError:
            # 无 numpy 时回退为单个连通分量
            return [list(nodes)]
        n = len(nodes)
        if n <= 1:
            return [list(nodes)]
        idx = {node: i for i, node in enumerate(nodes)}
        W = np.zeros((n, n))
        for a in nodes:
            for b in self._similarity_adj().get(a, ()):
                if b in idx:
                    W[idx[a], idx[b]] = self.simhash_similarity(a, b)
        deg = W.sum(axis=1)
        # 归一化拉普拉斯 L = I - D^{-1/2} W D^{-1/2}
        d_inv = np.diag(1.0 / np.sqrt(np.where(deg > 0, deg, 1.0)))
        L = np.eye(n) - d_inv @ W @ d_inv
        # 取前 k 小特征值对应的特征向量
        eigvals, eigvecs = np.linalg.eigh(L)
        if k is None:
            k = max(2, int(np.argmax(np.diff(eigvals)) + 1)) if n > 2 else 2
        k = max(2, min(k, n))
        feats = eigvecs[:, :k]
        # 简单 k-means（欧氏距离）
        rng = np.random.default_rng(0)
        centers = feats[rng.choice(n, k, replace=False)]
        labels = np.zeros(n, dtype=int)
        for _ in range(100):
            dist = ((feats[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
            new_labels = dist.argmin(axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for c in range(k):
                if (labels == c).any():
                    centers[c] = feats[labels == c].mean(axis=0)
        clusters = [[] for _ in range(k)]
        for i, node in enumerate(nodes):
            clusters[labels[i]].append(node)
        return [c for c in clusters if c]

    @staticmethod
    def group_centers(groups, H_map):
        """每群取 H 最大者为中心，返回 {groupId: center_Id}。"""
        centers = {}
        for gid, group in enumerate(groups):
            centers[gid] = max(group, key=lambda x: H_map.get(x, 0.0))
        return centers

    # ------------------------------------------------------------------
    # 传播影响力
    # ------------------------------------------------------------------
    def propagation_influence(self, nodes):
        """影响力 = tanh(N + log2(M) + min(1, log10(X/T)))。

        N: 来源平台数；M: 节点总数；X: 文本互动总量；T: 时间跨度（天）。
        """
        M = len(nodes)
        if M == 0:
            return 0.0
        platforms = set()
        X = 0
        times = []
        for Id in nodes:
            e = self.entities.get(Id)
            if e is None:
                continue
            if getattr(e, 'platform', None):
                platforms.add(e.platform)
            rd = getattr(e, 'reactiondata', None) or {}
            X += sum(v for v in rd.values() if isinstance(v, (int, float)))
            if getattr(e, 'time', None):
                times.append(_parse_time(e.time))
        N = len(platforms)
        # 时间跨度 T（天）：解析失败或单点视为 1 天，避免除零
        times = [t for t in times if t is not None]
        if len(times) >= 2:
            try:
                span = (max(times) - min(times)).total_seconds() / 86400.0
                T = max(span, 1.0)
            except (TypeError, ValueError):
                T = 1.0
        else:
            T = 1.0
        term = N + math.log2(max(M, 1)) + min(1.0, math.log10(max(X / T, 1e-9)))
        return math.tanh(term)

    # ------------------------------------------------------------------
    # 关联支撑度
    # ------------------------------------------------------------------
    def _support_contribution(self, rel):
        """单条边的支撑贡献。"""
        if rel.tp == 'topicsim':
            return -1.0 if rel.direction == 'opposite' else 0.6
        return SUPPORT_WEIGHT.get(rel.tp, 0.0)

    def support_degree(self, Id):
        """事件 B 受到的支撑度 S = 所有指向 B 的边的支撑贡献之和 + 额外证据补正。

        额外证据补正：对因果图中没有接入边的根节点 N，若 N->B 存在最短路且
        长度 D>1，则 S 增加 1/D。
        """
        S = sum(w for (src, dst, w) in self._directed_edges() if dst == Id)
        adj, indeg = self._causal_adj()
        for node, d in indeg.items():
            if d == 0 and node != Id:  # 根节点 N（无接入边）
                D = self._shortest_path(adj, node, Id)
                if D is not None and D > 1:
                    S += 1.0 / D
        return S

    # ------------------------------------------------------------------
    # 多源交叉印证与归一化支撑度
    # ------------------------------------------------------------------
    @staticmethod
    def _percentile(values, p):
        """返回第 p 百分位（0~100）。"""
        vals = sorted(values)
        if not vals:
            return 0.0
        k = (len(vals) - 1) * p / 100.0
        lo, hi = int(math.floor(k)), int(math.ceil(k))
        if lo == hi:
            return vals[lo]
        frac = k - lo
        return vals[lo] * (1 - frac) + vals[hi] * frac

    def cross_validation_degree(self, center_Id, group, H_map):
        """中心节点 C 的多源交叉印证度 Q = 1 + Σ W_i。

        W_i 为第 i 个（非 C 自身）信源中满足以下条件的文本数：
          - 与 C 的相似度在 0.5 ~ 0.8；
          - 与 C 存在观点方向大致相同的共主题边或因果边；
          - 其 H 位于群内所有节点 H 的上 35% 分位点之后。
        """
        center = self.entities.get(center_Id)
        if center is None:
            return 1.0
        center_source = getattr(center, 'platform', None)
        # 与 C 相邻的「共主题一致边」或「因果边」的对方节点
        supported = set()
        for r in self.relations:
            other = None
            if r.tp == 'reason' and r.Id1 == center_Id:
                other = r.Id2
            elif r.tp == 'topicsim' and r.direction != 'opposite':
                if r.Id1 == center_Id:
                    other = r.Id2
                elif r.Id2 == center_Id:
                    other = r.Id1
            if other is not None:
                supported.add(other)
        # H 的上 35% 分位点（即 65 分位）
        Hs = [H_map.get(x, 0.0) for x in group]
        thr = self._percentile(Hs, 65)
        per_source = defaultdict(int)
        for other_Id in group:
            if other_Id == center_Id:
                continue
            other = self.entities.get(other_Id)
            if other is None or getattr(other, 'platform', None) == center_source:
                continue
            sim = self.similarity(center_Id, other_Id)
            ok_sim = 0.5 <= sim <= 0.8
            ok_edge = other_Id in supported
            ok_H = H_map.get(other_Id, 0.0) >= thr
            if ok_sim and ok_edge and ok_H:
                per_source[getattr(other, 'platform', None)] += 1
        return 1.0 + sum(per_source.values())

    def normalized_support(self, center_Id, group, H_map):
        """归一化支撑度 S_r = sigmoid(Q + S)。"""
        Q = self.cross_validation_degree(center_Id, group, H_map)
        S = self.support_degree(center_Id)
        return 1.0 / (1.0 + math.exp(-(Q + S)))

    # ------------------------------------------------------------------
    # 价值判定
    # ------------------------------------------------------------------
    def value_H(self, entity, entropy=None, fmatch=0.0, kwmatch=0.0,
                candidates=None):
        """H = M * DS(T) * (1 - Sim(T))。

        M   = need_match_flag(fmatch, kwmatch)
        DS  = 信息密度 = 熵 / 文本长度
        Sim = 冗余度（其他文本 sim 的最大值）
        fmatch / kwmatch 需由外部（H_func.py 的领域分类器与关键词匹配）提供。
        """
        e = entity if isinstance(entity, Entity) else self.entities.get(entity)
        if e is None:
            return 0.0
        if entropy is None:
            entropy = char_entropy(e.text)
        DS = information_density(entropy, len(e.text))
        M = need_match_flag(fmatch, kwmatch)
        Sim = self.redundancy(e, candidates)
        return M * DS * (1.0 - Sim)

    # ------------------------------------------------------------------
    # 序列化
    # ------------------------------------------------------------------
    def to_dict(self):
        return {
            "entities": [e.to_dict() for e in self.entities.values()],
            "relations": [r.to_dict() for r in self.relations],
            "triples": [(h, r.to_dict(), t) for (h, r, t) in self.triples],
        }

    @classmethod
    def from_dict(cls, d):
        kg = cls()
        for ed in d.get("entities", []):
            if "platform" in ed:
                kg.add_entity(EventEntity.from_dict(ed))
            else:
                kg.add_entity(Entity.from_dict(ed))
        for rd in d.get("relations", []):
            kg.add_relation(Relation.from_dict(rd))
        for (h, rd, t) in d.get("triples", []):
            kg.triples.append((h, Relation.from_dict(rd), t))
        return kg

    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, 'r', encoding='utf-8') as f:
            return cls.from_dict(json.load(f))


# ---------------------------------------------------------------------------
# 独立工具函数
# ---------------------------------------------------------------------------

def _to_serializable(value):
    """把 datetime 等对象转为可 JSON 序列化的形式。"""
    if hasattr(value, 'isoformat'):
        return value.isoformat()
    return value


def _parse_time(value):
    """把时间字段归一化为 datetime（原样保留 None / 非时间值）。"""
    if value is None or hasattr(value, 'isoformat'):
        return value
    from datetime import datetime
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return value


def char_entropy(text):
    """基于字符二元组的香农熵（轻量实现，可替换为 H_func.entropy）。"""
    t = ''.join(text.split())
    if len(t) < 2:
        return 0.0
    freq = defaultdict(int)
    for i in range(len(t) - 1):
        freq[t[i:i + 2]] += 1
    total = sum(freq.values())
    entr = 0.0
    for c in freq.values():
        p = c / total
        entr -= p * math.log2(p)
    return entr


def information_density(entropy, length):
    """信息密度 DS = 熵 / 文本长度。"""
    return entropy / length if length > 0 else 0.0


def need_match_flag(fmatch, kwmatch, threshold=0.75):
    """需求契合标志 M = (FMatch>0.75 && KWMatch>0.75) -> {0,1}。"""
    return 1 if (fmatch > threshold and kwmatch > threshold) else 0


# ---------------------------------------------------------------------------
# 演示
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 构造一个含 5 个事件实体的示例图谱
    kg = KnowledgeGraph()
    events = [
        EventEntity(1, "高考加分政策再遭削减，教育局发布最新文件", "微博",
                    {"comments": 120, "forwards": 40}, "高考", time=__import__('datetime').datetime(2024, 6, 1)),
        EventEntity(2, "高考加分政策被削减，教育局公布新文件", "知乎",
                    {"comments": 80, "forwards": 25}, "高考", time=__import__('datetime').datetime(2024, 6, 2)),
        EventEntity(3, "高考加分再度收紧，教育部门公布文件说明", "头条",
                    {"comments": 60, "forwards": 15}, "高考", time=__import__('datetime').datetime(2024, 6, 3)),
        EventEntity(4, "某明星官宣恋情引发热议", "微博",
                    {"comments": 500, "forwards": 200}, "娱乐", time=__import__('datetime').datetime(2024, 6, 2)),
        EventEntity(5, "高考加分政策调整引发争议", "贴吧",
                    {"comments": 45, "forwards": 10}, "高考", time=__import__('datetime').datetime(2024, 6, 4)),
    ]
    for e in events:
        kg.add_entity(e)

    # 关系：1/2/3/5 为同一事件的相似/共主题表述，4 为无关事件
    kg.add_relation(Relation(1, 2, 'similarity'))
    kg.add_relation(Relation(1, 3, 'similarity'))
    kg.add_relation(Relation(2, 3, 'topicsim', direction='same'))
    kg.add_relation(Relation(1, 5, 'topicsim', direction='same'))
    kg.add_relation(Relation(1, 2, 'reason'))   # 1 因果支撑 2
    kg.add_relation(Relation(1, 3, 'time'))     # 1 先于 3

    print("== 相似度（Hash 层） ==")
    print("simhash(1,2) =", round(kg.simhash_similarity(1, 2), 4))
    print("simhash(1,4) =", round(kg.simhash_similarity(1, 4), 4))

    print("\n== 相似度（含图谱修正层） ==")
    print("similarity(1,2) =", round(kg.similarity(1, 2), 4))
    print("similarity(1,5) =", round(kg.similarity(1, 5), 4))

    print("\n== 冗余度 Sim ==")
    print("Sim(1) =", round(kg.redundancy(1), 4))

    print("\n== 相似事件群 ==")
    groups = kg.similar_event_groups()
    for g in groups:
        print("  群:", g)

    print("\n== 传播影响力 ==")
    for g in groups:
        print(f"  群{g} 影响力 =", round(kg.propagation_influence(g), 4))

    print("\n== 关联支撑度 S ==")
    for Id in [1, 2, 3]:
        print(f"  S({Id}) =", round(kg.support_degree(Id), 4))

    print("\n== 多源交叉印证 Q / 归一化支撑度 S_r ==")
    H_map = {1: 3.0, 2: 2.4, 3: 2.1, 4: 1.0, 5: 1.5}
    for g in groups:
        c = kg.group_centers([g], H_map)[0]
        print(f"  群{g} 中心={c}，Q={round(kg.cross_validation_degree(c, g, H_map), 4)}，"
              f"S_r={round(kg.normalized_support(c, g, H_map), 4)}")

    print("\n== 价值判定 H ==")
    for Id in [1, 4]:
        h = kg.value_H(Id, fmatch=0.9, kwmatch=0.9)
        print(f"  H({Id}) =", round(h, 4))

    print("\n== 持久化 ==")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kg_demo.json")
    kg.save(out)
    kg2 = KnowledgeGraph.load(out)
    print(f"  保存/加载后实体数 = {len(kg2)}，文件 = {out}")
