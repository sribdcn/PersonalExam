# -*- coding: utf-8 -*-
"""
Copyright (c) 2025 AI系统及应用课题组@SRIBD

基于LLM和知识图谱协同的个性化出题系统 (PersonalExam)
Personalized Question Generation System Based on LLM and Knowledge Graph Collaboration

智能题目选择器 - 基于RAG和知识图谱
使用盘古7B从检索结果中选择最合适的题目
"""

import logging
import json
import re
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class SmartQuestionSelector:
    """智能题目选择器"""
    
    def __init__(self, rag_engine, llm_model, question_db):
        """
        初始化选择器
        
        Args:
            rag_engine: RAG引擎
            llm_model: 盘古7B模型
            question_db: 题库数据库
        """
        self.rag_engine = rag_engine
        self.llm_model = llm_model
        self.question_db = question_db
        
        logger.info("✅ 智能题目选择器初始化完成")
    
    def select_question(self, 
                       student_id: str,
                       student_mastery: float,
                       major_point: str,
                       minor_point: str,
                       used_question_ids: set,
                       top_k: int = 5) -> Optional[Dict[str, Any]]:
        """
        选择最合适的题目（带多级降级策略）
        
        Args:
            student_id: 学生ID
            student_mastery: 学生掌握度
            major_point: 知识点大类
            minor_point: 知识点小类
            used_question_ids: 已使用的题目ID
            top_k: 检索题目数量
            
        Returns:
            选中的题目
        """
        logger.info(f"🎯 为学生 {student_id} 选择题目: {major_point}/{minor_point}, "
                   f"掌握度 {student_mastery:.3f}")
        
        # 1. 构建知识子图
        subgraph = self.rag_engine.build_knowledge_subgraph(
            student_mastery=student_mastery,
            major_point=major_point,
            minor_point=minor_point,
            top_k=top_k
        )
        
        if not subgraph['retrieved_questions']:
            logger.warning("⚠️ RAG未检索到题目，使用多级降级备用方案")
            return self._multi_level_fallback_selection(
                major_point, minor_point, student_mastery, used_question_ids
            )
        
        # 2. 过滤掉已使用的题目
        candidate_questions = []
        for item in subgraph['retrieved_questions']:
            q = item['question']
            q_id = q.get('题号')
            if q_id not in used_question_ids:
                candidate_questions.append(item)
        
        if not candidate_questions:
            logger.warning(f"⚠️ RAG检索到的 {len(subgraph['retrieved_questions'])} 道题都已使用，"
                          f"使用多级降级备用方案")
            return self._multi_level_fallback_selection(
                major_point, minor_point, student_mastery, used_question_ids
            )
        
        # 3. 使用盘古7B选择最合适的题目
        selected_question = self._llm_select_question(
            candidate_questions=candidate_questions,
            student_mastery=student_mastery,
            knowledge_subgraph=subgraph
        )
        
        if selected_question:
            logger.info(f"✅ 选中题目 {selected_question.get('题号')} "
                       f"(难度: {selected_question.get('难度', 0.5):.2f})")
            return selected_question
        else:
            # 如果LLM选择失败，使用简单策略
            logger.warning("⚠️ LLM选择失败，使用启发式策略")
            return candidate_questions[0]['question']
    
    def _llm_select_question(self,
                            candidate_questions: List[Dict[str, Any]],
                            student_mastery: float,
                            knowledge_subgraph: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        使用盘古7B选择题目
        
        Args:
            candidate_questions: 候选题目列表
            student_mastery: 学生掌握度
            knowledge_subgraph: 知识子图
            
        Returns:
            选中的题目
        """
        # 如果只有1道候选题，直接返回
        if len(candidate_questions) == 1:
            logger.info("✅ 只有1道候选题，直接选择")
            return candidate_questions[0]['question']
        
        # 构建简洁的候选题目列表
        candidates_text = ""
        for i, item in enumerate(candidate_questions, 1):
            q = item['question']
            candidates_text += f"""题目{i} (ID:{q.get('题号')}, 难度:{q.get('难度', 0.5):.2f}, 相似度:{item.get('score', 0):.3f})
问题: {q.get('问题', '')[:80]}...
"""
        
        # 简化知识图谱信息
        entities_text = "、".join([e['name'] for e in knowledge_subgraph['entities'][:5]]) if knowledge_subgraph['entities'] else "无"
        
        # 优化提示词 - 更简洁清晰
        prompt = f"""你是数学教师，为学生选择最合适的题目。

学生情况: 掌握度{student_mastery:.1%}，目标知识点{knowledge_subgraph['target_knowledge']}
相关概念: {entities_text}

候选题目（共{len(candidate_questions)}道）:
{candidates_text}

要求: 选择1道最适合该学生当前水平的题目

输出格式(只输出数字):
ID: [题目ID数字]
"""
        
        try:
            # 确保模型已加载
            if not self.llm_model.is_loaded:
                logger.info("🔄 加载盘古7B模型...")
                self.llm_model.load_model()
            
            # 生成（优化参数以提升速度）
            logger.info("🤖 盘古7B正在选择题目...")
            response = self.llm_model.generate(
                prompt, 
                temperature=0.2,  # 降低温度，提升速度和稳定性
                max_length=64,  # 大幅缩短生成长度（只需要ID数字）
                enable_thinking=False  # 关闭思维链，提升速度
            )
            
            # 解析响应
            selected_id = self._parse_selection_response_simple(response)
            
            if selected_id is None:
                logger.warning("⚠️ LLM选择失败，使用启发式规则")
                return self._heuristic_selection(candidate_questions, student_mastery)
            
            # 查找对应题目
            for item in candidate_questions:
                if item['question'].get('题号') == selected_id:
                    logger.info(f"✅ 盘古7B选中题目 {selected_id}")
                    return item['question']
            
            logger.warning(f"⚠️ 选中的ID {selected_id} 不存在，使用启发式规则")
            return self._heuristic_selection(candidate_questions, student_mastery)
            
        except Exception as e:
            logger.error(f"❌ LLM选择失败: {e}")
            return self._heuristic_selection(candidate_questions, student_mastery)
    
    def _parse_selection_response_simple(self, response: str) -> Optional[int]:
        """简化的响应解析 - 只提取数字ID"""
        try:
            # 方法1: 查找 "ID: 数字" 模式
            patterns = [
                r'ID\s*[：:]\s*(\d+)',
                r'题目\s*(\d+)',
                r'选择\s*(\d+)',
                r'(\d+)',  # 任何数字
            ]
            
            for pattern in patterns:
                match = re.search(pattern, response)
                if match:
                    selected_id = int(match.group(1))
                    logger.debug(f"提取到ID: {selected_id}")
                    return selected_id
            
            return None
            
        except Exception as e:
            logger.warning(f"⚠️ 解析失败: {e}")
            return None
    
    def _heuristic_selection(self, 
                            candidate_questions: List[Dict[str, Any]],
                            student_mastery: float) -> Dict[str, Any]:
        """
        启发式题目选择（后备方案）
        综合考虑：难度匹配 + RAG相似度
        """
        logger.info("📊 使用启发式规则选择题目")
        
        # 根据掌握度确定目标难度
        if student_mastery < 0.3:
            target_difficulty = 0.25  # 简单
        elif student_mastery < 0.7:
            target_difficulty = 0.50  # 中等
        else:
            target_difficulty = 0.75  # 困难
        
        # 计算每道题的综合得分
        best_question = None
        best_score = -999999
        
        for item in candidate_questions:
            q = item['question']
            difficulty = q.get('难度', 0.5)
            rag_score = item.get('score', 0)
            
            # 难度匹配得分（越接近目标越好）
            difficulty_score = 1.0 - abs(difficulty - target_difficulty)
            
            # 综合得分 = 难度匹配(60%) + RAG相似度(40%)
            total_score = 0.6 * difficulty_score + 0.4 * rag_score
            
            if total_score > best_score:
                best_score = total_score
                best_question = q
        
        if best_question:
            logger.info(f"✅ 启发式选中题目 {best_question.get('题号')} (得分: {best_score:.3f})")
        
        return best_question
    
    def _multi_level_fallback_selection(self, 
                                       major_point: str,
                                       minor_point: str,
                                       student_mastery: float,
                                       used_question_ids: set) -> Optional[Dict[str, Any]]:
        """
        多级降级备用选择方案
        
        降级策略：
        1. 精确匹配：major + minor + difficulty
        2. 大类匹配：major + difficulty
        3. 大类匹配：major
        4. 最后备用：任意未使用的题目
        
        Args:
            major_point: 知识点大类
            minor_point: 知识点小类
            student_mastery: 学生掌握度
            used_question_ids: 已使用的题目ID
            
        Returns:
            选中的题目
        """
        logger.info("🔄 启动多级降级备用选择方案...")
        
        # 根据掌握度确定难度范围
        if student_mastery < 0.3:
            difficulty_range = (0.0, 0.4)
            difficulty_desc = "简单"
        elif student_mastery < 0.7:
            difficulty_range = (0.3, 0.7)
            difficulty_desc = "中等"
        else:
            difficulty_range = (0.6, 1.0)
            difficulty_desc = "困难"
        
        # 【第1级】精确匹配：major + minor + difficulty
        logger.info(f"📍 第1级：尝试精确匹配（{major_point}/{minor_point}, 难度{difficulty_desc}）")
        questions = self.question_db.get_questions_filtered(
            major_point=major_point,
            minor_point=minor_point,
            difficulty_range=difficulty_range
        )
        available = [q for q in questions if q.get('题号') not in used_question_ids]
        
        if available:
            import random
            selected = random.choice(available)
            logger.info(f"✅ 第1级成功：选中题目 {selected.get('题号')} "
                       f"({major_point}/{minor_point}, 难度{selected.get('难度', 0.5):.2f})")
            return selected
        else:
            logger.info(f"⚠️  第1级失败：{major_point}/{minor_point} + 难度范围{difficulty_range}下无可用题目")
        
        # 【第2级】大类匹配：major + difficulty（忽略minor）
        logger.info(f"📍 第2级：尝试同大类其他小类（{major_point}, 难度{difficulty_desc}）")
        questions = self.question_db.get_questions_filtered(
            major_point=major_point,
            minor_point=None,  # 不限制小类
            difficulty_range=difficulty_range
        )
        available = [q for q in questions if q.get('题号') not in used_question_ids]
        
        if available:
            import random
            selected = random.choice(available)
            selected_minor = selected.get('知识点小类', selected.get('knowledge_point_minor', '未知'))
            logger.info(f"✅ 第2级成功：选中题目 {selected.get('题号')} "
                       f"({major_point}/{selected_minor}, 难度{selected.get('难度', 0.5):.2f})")
            return selected
        else:
            logger.info(f"⚠️  第2级失败：{major_point}大类 + 难度范围{difficulty_range}下无可用题目")
        
        # 【第3级】大类匹配：major（忽略difficulty）
        logger.info(f"📍 第3级：尝试同大类任意难度（{major_point}）")
        questions = self.question_db.get_questions_filtered(
            major_point=major_point,
            minor_point=None,
            difficulty_range=None  # 不限制难度
        )
        available = [q for q in questions if q.get('题号') not in used_question_ids]
        
        if available:
            import random
            selected = random.choice(available)
            selected_minor = selected.get('知识点小类', selected.get('knowledge_point_minor', '未知'))
            logger.info(f"✅ 第3级成功：选中题目 {selected.get('题号')} "
                       f"({major_point}/{selected_minor}, 难度{selected.get('难度', 0.5):.2f})")
            return selected
        else:
            logger.info(f"⚠️  第3级失败：{major_point}大类下无可用题目")
        
        # 【第4级】最后备用：任意未使用的题目
        logger.warning("📍 第4级：选择任意未使用的题目（不限知识点和难度）")
        all_questions = self.question_db.get_all_questions()
        available = [q for q in all_questions if q.get('题号') not in used_question_ids]
        
        if available:
            import random
            selected = random.choice(available)
            selected_major = selected.get('知识点大类', selected.get('knowledge_point_major', '未知'))
            selected_minor = selected.get('知识点小类', selected.get('knowledge_point_minor', '未知'))
            logger.warning(f"⚠️  第4级成功（但偏离目标）：选中题目 {selected.get('题号')} "
                          f"({selected_major}/{selected_minor}, 难度{selected.get('难度', 0.5):.2f})")
            return selected
        else:
            # 真的没题了
            logger.error("❌ 所有4级备用方案均失败：题库中所有题目都已使用或无可用题目")
            total_count = len(all_questions)
            used_count = len(used_question_ids)
            logger.error(f"📊 题库统计：总题目{total_count}道，已使用{used_count}道，"
                        f"剩余{total_count - used_count}道")
            return None


def create_question_selector(rag_engine, llm_model, question_db) -> SmartQuestionSelector:
    """创建题目选择器"""
    return SmartQuestionSelector(rag_engine, llm_model, question_db)


if __name__ == "__main__":
    # 测试代码
    import sys
    sys.path.append("..")
    from config import (BGE_M3_MODEL_PATH, PANGU_MODEL_PATH, 
                       EMBEDDING_MODEL_CONFIG, PANGU_MODEL_CONFIG, QUESTION_DB)
    from models.embedding_model import create_embedding_model
    from models.llm_models import create_llm_model
    from data_management.question_db import create_question_database
    from knowledge_management.rag_engine import create_rag_engine
    
    logging.basicConfig(level=logging.INFO)
    
    # 创建组件
    embedding_model = create_embedding_model(BGE_M3_MODEL_PATH, EMBEDDING_MODEL_CONFIG)
    llm_model = create_llm_model('pangu', PANGU_MODEL_PATH, PANGU_MODEL_CONFIG)
    question_db = create_question_database(str(QUESTION_DB))
    rag_engine = create_rag_engine(embedding_model, llm_model)
    
    # 构建索引
    all_questions = question_db.get_all_questions()
    rag_engine.build_question_index(all_questions)
    
    # 创建选择器
    selector = create_question_selector(rag_engine, llm_model, question_db)
    
    # 测试选择
    selected = selector.select_question(
        student_id="test_001",
        student_mastery=0.5,
        major_point="代数",
        minor_point="一元二次方程",
        used_question_ids=set()
    )
    
    if selected:
        print(f"选中题目: {selected.get('题号')}")
    else:
        print("选择失败")