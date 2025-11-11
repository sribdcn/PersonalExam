# -*- coding: utf-8 -*-
"""
Copyright (c) 2025 AI系统及应用课题组@SRIBD

Personalized Question Generation System Based on LLM and Knowledge Graph Collaboration

本地RAG引擎 - 基于向量检索和知识图谱
"""

import logging
import numpy as np
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import re

logger = logging.getLogger(__name__)


class LocalRAGEngine:
    """本地RAG引擎 - 向量检索 + 知识图谱"""
    
    def __init__(self, embedding_model, llm_model):
        """
        
        Args:
            embedding_model: BGE嵌入模型
            llm_model: 盘古7B模型
        """
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        
        # 题目索引
        self.question_texts = []  # 题目文本列表
        self.question_embeddings = None  # 题目嵌入矩阵
        self.question_metadata = []  # 题目元数据
        
        logger.info("✅ 本地RAG引擎初始化完成")
    
    def build_question_index(self, questions: List[Dict[str, Any]]):

        logger.info(f"🔄 正在为 {len(questions)} 道题目构建向量索引...")
        
        self.question_texts = []
        self.question_metadata = []
        
        for q in questions:
            # 构建题目的文本表示
            text = self._format_question_for_indexing(q)
            self.question_texts.append(text)
            
            # 保存元数据
            self.question_metadata.append({
                'question': q,
                'major_point': q.get('知识点大类', ''),
                'minor_point': q.get('知识点小类', ''),
                'difficulty': q.get('难度', 0.5),
                'id': q.get('题号', len(self.question_metadata))
            })
        
        # 批量计算嵌入
        logger.info("🔄 正在计算题目嵌入...")
        self.question_embeddings = self.embedding_model.encode(
            self.question_texts,
            normalize=True
        )
        
        logger.info(f"✅ 题目索引构建完成: {len(self.question_texts)} 道题, "
                   f"嵌入维度 {self.question_embeddings.shape[1]}")
    
    def _format_question_for_indexing(self, question: Dict[str, Any]) -> str:
        """格式化题目用于索引"""
        major = question.get('知识点大类', '')
        minor = question.get('知识点小类', '')
        problem = question.get('问题', '')
        answer = question.get('答案', '')
        explanation = question.get('解析', '')
        
        # 组合关键信息
        text = f"知识点：{major} {minor}\n问题：{problem}\n答案：{answer}\n解析：{explanation}"
        return text
    
    def search_questions(self, query: str, 
                        major_point: Optional[str] = None,
                        minor_point: Optional[str] = None,
                        difficulty_range: Optional[Tuple[float, float]] = None,
                        top_k: int = 5) -> List[Dict[str, Any]]:

        if self.question_embeddings is None:
            logger.error("❌ 题目索引未构建")
            return []
        
        # 计算查询嵌入
        query_embedding = self.embedding_model.encode([query], normalize=True)[0]
        
        # 计算相似度
        similarities = np.dot(self.question_embeddings, query_embedding)
        
        # 获取候选题目
        candidates = []
        for idx, score in enumerate(similarities):
            metadata = self.question_metadata[idx]
            
            # 应用过滤条件
            if major_point and metadata['major_point'] != major_point:
                continue
            if minor_point and metadata['minor_point'] != minor_point:
                continue
            if difficulty_range:
                diff = metadata['difficulty']
                if not (difficulty_range[0] <= diff < difficulty_range[1]):
                    continue
            
            candidates.append({
                'question': metadata['question'],
                'score': float(score),
                'metadata': metadata
            })
        
        # 按相似度排序
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # 返回top_k
        results = candidates[:top_k]
        
        logger.info(f"🔍 检索到 {len(results)} 道相关题目 (query: '{query[:50]}...')")
        return results
    
    def extract_entities_and_relations(self, text_context: str) -> Dict[str, Any]:

        prompt = f"""分析以下数学题目，提取关键的知识点实体。

题目内容：
{text_context[:1000]}

要求：
1. 提取3-5个核心数学知识点
2. 每个知识点用一个词或短语表示
3. 严格按照以下格式输出，必须是有效的JSON：

{{
  "entities": [
    {{"name": "一元二次方程", "type": "知识点"}},
    {{"name": "因式分解", "type": "方法"}}
  ],
  "relations": [
    {{"source": "一元二次方程", "target": "因式分解", "relation": "可以使用"}}
  ]
}}

只输出JSON，不要有任何解释文字。
"""
        
        try:
            # 确保盘古7B已加载
            if not self.llm_model.is_loaded:
                logger.info("🔄 加载盘古7B模型...")
                self.llm_model.load_model()
            
            # 生成（降低温度以获得更稳定的JSON）
            response = self.llm_model.generate(prompt, temperature=0.1, max_length=1024)
            
            # 解析JSON
            kg_data = self._parse_kg_response(response)
            
            logger.info(f"✅ 提取到 {len(kg_data.get('entities', []))} 个实体, "
                       f"{len(kg_data.get('relations', []))} 个关系")
            
            return kg_data
            
        except Exception as e:
            logger.error(f"❌ 实体关系提取失败: {e}")
            return {'entities': [], 'relations': []}
    
    def _parse_kg_response(self, response: str) -> Dict[str, Any]:
        """解析知识图谱响应"""
        try:
            # 查找JSON部分
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                return {'entities': [], 'relations': []}
            
            json_str = response[start_idx:end_idx]
            kg_data = json.loads(json_str)
            
            return kg_data
        except Exception as e:
            logger.warning(f"⚠️ JSON解析失败: {e}")
            
            # 尝试正则提取
            entities = []
            relations = []
            
            # 提取实体
            entity_pattern = r'实体[:：]\s*([^\n]+)'
            for match in re.finditer(entity_pattern, response):
                entities.append({'name': match.group(1).strip(), 'type': '概念'})
            
            # 提取关系
            relation_pattern = r'关系[:：]\s*([^\n]+)'
            for match in re.finditer(relation_pattern, response):
                relations.append({'source': '', 'target': '', 'relation': match.group(1).strip()})
            
            return {'entities': entities, 'relations': relations}
    
    def build_knowledge_subgraph(self, 
                                student_mastery: float,
                                major_point: str,
                                minor_point: str,
                                top_k: int = 5) -> Dict[str, Any]:
        """
        构建知识子图
        
        Args:
            student_mastery: 学生掌握度
            major_point: 知识点大类
            minor_point: 知识点小类
            top_k: 检索题目数量
            
        Returns:
            知识子图数据
        """
        logger.info(f"🔄 构建知识子图: {major_point}/{minor_point}, 掌握度: {student_mastery:.3f}")
        
        # 1. 构建查询
        if student_mastery < 0.3:
            difficulty_desc = "简单 基础"
        elif student_mastery < 0.7:
            difficulty_desc = "中等"
        else:
            difficulty_desc = "困难 提高"
        
        query = f"{major_point} {minor_point} {difficulty_desc}"
        
        # 2. 检索相关题目
        retrieved_questions = self.search_questions(
            query=query,
            major_point=major_point,
            minor_point=minor_point,
            top_k=top_k
        )
        
        if not retrieved_questions:
            logger.warning("⚠️ 未检索到相关题目")
            return {
                'retrieved_questions': [],
                'entities': [],
                'relations': [],
                'context': ''
            }
        
        # 3. 构建上下文
        context_texts = []
        for item in retrieved_questions:
            q = item['question']
            text = f"""题目{q.get('题号', '')}:
知识点: {q.get('知识点大类', '')} / {q.get('知识点小类', '')}
难度: {q.get('难度', 0.5)}
问题: {q.get('问题', '')}
答案: {q.get('答案', '')}
解析: {q.get('解析', '')}
"""
            context_texts.append(text)
        
        full_context = "\n\n".join(context_texts)
        
        # 4. 提取实体和关系
        kg_data = self.extract_entities_and_relations(full_context)
        
        # 5. 组合结果
        subgraph = {
            'retrieved_questions': retrieved_questions,
            'entities': kg_data.get('entities', []),
            'relations': kg_data.get('relations', []),
            'context': full_context,
            'student_mastery': student_mastery,
            'target_knowledge': f"{major_point}/{minor_point}"
        }
        
        logger.info(f"✅ 知识子图构建完成: {len(retrieved_questions)} 道题, "
                   f"{len(subgraph['entities'])} 个实体")
        
        return subgraph
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取索引统计信息"""
        return {
            'total_questions': len(self.question_texts),
            'embedding_dim': self.question_embeddings.shape[1] if self.question_embeddings is not None else 0,
            'indexed': self.question_embeddings is not None
        }


def create_rag_engine(embedding_model, llm_model) -> LocalRAGEngine:
    """创建RAG引擎"""
    return LocalRAGEngine(embedding_model, llm_model)


if __name__ == "__main__":
    # 测试代码
    import sys
    sys.path.append("..")
    from config import BGE_M3_MODEL_PATH, PANGU_MODEL_PATH, EMBEDDING_MODEL_CONFIG, PANGU_MODEL_CONFIG
    from models.embedding_model import create_embedding_model
    from models.llm_models import create_llm_model
    
    logging.basicConfig(level=logging.INFO)
    
    # 创建模型
    embedding_model = create_embedding_model(BGE_M3_MODEL_PATH, EMBEDDING_MODEL_CONFIG)
    llm_model = create_llm_model('pangu', PANGU_MODEL_PATH, PANGU_MODEL_CONFIG)
    
    # 创建RAG引擎
    rag = create_rag_engine(embedding_model, llm_model)
    
    # 测试题目
    test_questions = [
        {
            '题号': 1,
            '问题': 'x^2 - 5x + 6 = 0',
            '答案': 'x = 2 或 x = 3',
            '难度': 0.3,
            '知识点大类': '代数',
            '知识点小类': '一元二次方程',
            '解析': '因式分解得 (x-2)(x-3)=0'
        }
    ]
    
    # 构建索引
    rag.build_question_index(test_questions)
    
    # 测试检索
    results = rag.search_questions("二次方程", major_point="代数", top_k=3)
    print(f"检索结果: {len(results)} 道题")
    
    # 测试知识子图
    subgraph = rag.build_knowledge_subgraph(0.5, "代数", "一元二次方程")
    print(f"知识子图: {len(subgraph['entities'])} 个实体")