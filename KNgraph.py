#知识图谱
class entity:
    def __init__(self,Id,text):#实体父类
        self.Id=Id
        self.text=text
class relation:
    def __init__(self,Id1,Id2,tp):#关系父类
        self.Id1=Id1
        self.Id2=Id2
        self.tp=tp
class KnowledgeGraph:#图谱类
    def __init__(self):
        self.entities = set()  # 实体集合
        self.relations = set()  # 关系集合
        self.triples = []  # 三元组列表
    def add_entity(self, entity):
        """添加实体"""
        self.entities.add(entity)
    def add_relation(self, relation):
        """添加关系"""
        self.relations.add(relation)
    def add_triple(self, head, relation, tail):
        """添加三元组"""
        if head in self.entities and tail in self.entities and relation in self.relations:
            self.triples.append((head, relation, tail))
        else:
            raise ValueError("实体或关系不存在")
    def get_entities(self):
        """获取所有实体"""
        return list(self.entities)
    def get_relations(self):
        """获取所有关系"""
        return list(self.relations)
    def get_triples(self):
        """获取所有三元组"""
        return self.triples
TPlist=['reason','time','similarity','topicsim']
class evententity(entity):
    def __init__(self,Id,text,platform,reactiondata,topic):
        super().__init__(Id,text)
        self.platform=platform
        self.reactiondata=reactiondata#所有互动的数据表
        self.topic=topic