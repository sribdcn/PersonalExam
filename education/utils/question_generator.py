# -*- coding: utf-8 -*-
"""
智能题目选择器 - 基于知识图谱RAG
简化版：只负责调用RAG和备用策略
"""

import logging
import random
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class SmartQuestionSelector:
    """智能题目选择器"""
    
    def __init__(self, kg_rag, question_db, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            kg_rag: 知识图谱RAG引擎
            question_db: 题库数据库
            config: 配置
        """
        self.kg_rag = kg_rag
        self.question_db = question_db
        self.config = config or {}
        
        logger.info("✅ 智能题目选择器初始化完成")
    
    def select_question(self,
                       student_id: str,
                       student_mastery: float,
                       major_point: str,
                       minor_point: str,
                       used_question_ids: set,
                       top_k: int = 5) -> Optional[Dict[str, Any]]:
        """
        选择最适合的题目
        
        优先级：
        1. 使用知识图谱RAG检索
        2. 多级备用方案（SQL检索）
        
        Args:
            student_id: 学生ID
            student_mastery: 学生掌握度
            major_point: 知识点大类
            minor_point: 知识点小类
            used_question_ids: 已使用的题目ID
            top_k: RAG检索题目数量
            
        Returns:
            选中的题目
        """
        logger.info(f"🎯 为学生 {student_id} 选择题目: {major_point}/{minor_point}, "
                   f"掌握度 {student_mastery:.3f}")
        
        # 策略1：使用知识图谱RAG检索
        try:
            candidates = self.kg_rag.search_questions_for_student(
                student_id=student_id,
                major_point=major_point,
                minor_point=minor_point,
                student_mastery=student_mastery,
                used_question_ids=used_question_ids,
                top_k=top_k
            )
            
            if candidates:
                # 选择得分最高的
                selected = candidates[0]['question']
                logger.info(f"✅ RAG选中题目 {selected.get('题号')} "
                           f"(得分: {candidates[0]['score']:.3f})")
                return selected
        except Exception as e:
            logger.warning(f"⚠️  RAG检索失败: {e}")
        
        # 策略2：降级到多级SQL备用方案
        logger.info("📍 RAG未找到合适题目，使用多级备用方案...")
        return self._fallback_selection(
            major_point, minor_point, student_mastery, used_question_ids
        )
    
    def _fallback_selection(self,
                           major_point: str,
                           minor_point: str,
                           student_mastery: float,
                           used_question_ids: set) -> Optional[Dict[str, Any]]:
        """
        多级备用选择方案（纯SQL）
        
        级别：
        1. 精确匹配（major + minor + difficulty）
        2. 大类匹配（major + difficulty）
        3. 大类匹配（major，不限difficulty）
        4. 任意可用题目
        """
        # 确定难度范围
        if student_mastery < 0.3:
            difficulty_range = (0.0, 0.4)
        elif student_mastery < 0.7:
            difficulty_range = (0.3, 0.7)
        else:
            difficulty_range = (0.6, 1.0)
        
        # 级别1：精确匹配
        questions = self.question_db.get_questions_filtered(
            major_point=major_point,
            minor_point=minor_point,
            difficulty_range=difficulty_range
        )
        available = [q for q in questions if q.get('题号') not in used_question_ids]
        if available:
            selected = random.choice(available)
            logger.info(f"✅ 备用方案级别1成功: 题目 {selected.get('题号')}")
            return selected
        
        # 级别2：大类 + 难度
        questions = self.question_db.get_questions_filtered(
            major_point=major_point,
            minor_point=None,
            difficulty_range=difficulty_range
        )
        available = [q for q in questions if q.get('题号') not in used_question_ids]
        if available:
            selected = random.choice(available)
            logger.info(f"✅ 备用方案级别2成功: 题目 {selected.get('题号')}")
            return selected
        
        # 级别3：大类，不限难度
        questions = self.question_db.get_questions_filtered(
            major_point=major_point,
            minor_point=None,
            difficulty_range=None
        )
        available = [q for q in questions if q.get('题号') not in used_question_ids]
        if available:
            selected = random.choice(available)
            logger.info(f"✅ 备用方案级别3成功: 题目 {selected.get('题号')}")
            return selected
        
        # 级别4：任意题目
        all_questions = self.question_db.get_all_questions()
        available = [q for q in all_questions if q.get('题号') not in used_question_ids]
        if available:
            selected = random.choice(available)
            logger.warning(f"⚠️  备用方案级别4（偏离目标）: 题目 {selected.get('题号')}")
            return selected
        
        logger.error("❌ 所有备用方案均失败")
        return None


def create_question_selector(kg_rag, question_db, config: Optional[Dict[str, Any]] = None):
    """创建题目选择器"""
    return SmartQuestionSelector(kg_rag, question_db, config)