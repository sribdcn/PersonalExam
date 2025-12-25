# -*- coding: utf-8 -*-

import logging
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class BKTParameters:
    p_init: float = 0.3
    p_learn: float = 0.2
    p_guess: float = 0.3
    p_slip: float = 0.1
    p_forget: float = 0.05


class BKTDatabaseAdapter:
    
    def __init__(self, db_manager, default_params: Optional[BKTParameters] = None):

        self.db = db_manager
        self.default_params = default_params or BKTParameters()
        logger.info("✅ BKT数据库适配器初始化完成")
    
    def get_student_state(self, student_id: str, major_point: str, 
                         minor_point: str) -> Dict[str, Any]:
        """获取或初始化学生状态"""
        state = self.db.get_student_state(student_id, major_point, minor_point)
        
        if not state:
      
            params = self._get_personalized_params(student_id)
            self.db.upsert_student_state(
                student_id, major_point, minor_point,
                params.p_init,
                {
                    'p_init': params.p_init,
                    'p_learn': params.p_learn,
                    'p_guess': params.p_guess,
                    'p_slip': params.p_slip,
                    'p_forget': params.p_forget
                }
            )
            state = self.db.get_student_state(student_id, major_point, minor_point)
            logger.info(f"🆕 初始化学生状态: {student_id} - {major_point}/{minor_point}")
        
        return {
            'student_id': state['student_id'],
            'knowledge_point_major': state['major_point'],
            'knowledge_point_minor': state['minor_point'],
            'mastery_prob': state['mastery_prob'],
            'params': BKTParameters(
                p_init=state['p_init'],
                p_learn=state['p_learn'],
                p_guess=state['p_guess'],
                p_slip=state['p_slip'],
                p_forget=state['p_forget']
            )
        }
    
    def _get_personalized_params(self, student_id: str) -> BKTParameters:

        history = self.db.get_answer_history(student_id, limit=100)
        
        if len(history) < 10:
            return self.default_params

        correct = sum(1 for r in history if r['is_correct'])
        accuracy = correct / len(history)
        
        params = BKTParameters()
        
        if accuracy > 0.8:
            params.p_init = 0.5
        elif accuracy > 0.6:
            params.p_init = 0.4
        else:
            params.p_init = 0.2
        
        recent_10 = history[:10]
        recent_accuracy = sum(1 for r in recent_10 if r['is_correct']) / len(recent_10)
        
        if recent_accuracy > accuracy + 0.1:
            params.p_learn = 0.3  
        elif recent_accuracy > accuracy:
            params.p_learn = 0.2
        else:
            params.p_learn = 0.15
        
        return params
    
    def update_mastery_probability(self, state: Dict, is_correct: bool) -> float:
        p_mastery = state['mastery_prob']
        params = state['params']
        
        p_learn = params.p_learn
        p_forget = params.p_forget
        p_guess = params.p_guess
        p_slip = params.p_slip
        
        if is_correct:

            numerator = p_mastery * (1 - p_slip)
            denominator = numerator + (1 - p_mastery) * p_guess
            p_mastery_given_correct = numerator / denominator if denominator > 0 else p_mastery
            p_mastery_updated = p_mastery_given_correct + (1 - p_mastery_given_correct) * p_learn
        else:
 
            numerator = p_mastery * p_slip
            denominator = numerator + (1 - p_mastery) * (1 - p_guess)
            p_mastery_given_incorrect = numerator / denominator if denominator > 0 else p_mastery
            p_mastery_updated = p_mastery_given_incorrect * (1 - p_forget)
        

        p_mastery_updated = max(0.01, min(0.99, p_mastery_updated))
        
        return p_mastery_updated
    
    def record_answer(self, student_id: str, major_point: str, minor_point: str,
                     question: Dict[str, Any], is_correct: bool) -> Dict[str, Any]:
      
        state = self.get_student_state(student_id, major_point, minor_point)
        
        mastery_before = state['mastery_prob']
        
        mastery_after = self.update_mastery_probability(state, is_correct)
        
        self.db.upsert_student_state(
            student_id, major_point, minor_point, mastery_after
        )
        
        self.db.insert_answer_record(
            student_id,
            question.get('题号', 0),
            major_point,
            minor_point,
            '',
            is_correct,
            question.get('难度', 0.5),
            mastery_before,
            mastery_after
        )
        
        recent_perf = self.db.get_recent_performance(student_id, major_point, minor_point)
        recent_accuracy = sum(recent_perf) / len(recent_perf) if recent_perf else 0.0
        
        result = {
            'student_id': student_id,
            'knowledge_point_major': major_point,
            'knowledge_point_minor': minor_point,
            'current_mastery': mastery_after,
            'previous_mastery': mastery_before,
            'mastery_change': mastery_after - mastery_before,
            'total_answers': len(self.db.get_answer_history(student_id, major_point, minor_point)),
            'recent_accuracy': recent_accuracy
        }
        
        logger.info(f"📝 答题记录: {student_id} - {minor_point}: "
                   f"{mastery_before:.3f} → {mastery_after:.3f} "
                   f"({'✓' if is_correct else '✗'})")
        
        return result
    
    def get_weak_knowledge_points(self, student_id: str, 
                                  threshold: float = 0.4) -> List[Tuple[str, str, float]]:
        """获取薄弱知识点"""
        return self.db.get_weak_points(student_id, threshold)
    
    def get_all_mastery_status(self, student_id: str) -> Dict[str, Dict[str, float]]:
        """获取所有知识点掌握情况"""
        return self.db.get_student_all_states(student_id)
    
    def generate_student_profile(self, student_id: str) -> Dict[str, Any]:
        """
        生成/获取学生档案
        (修复：统一方法名为 generate_student_profile)
        """
        profile = self.db.get_student_profile(student_id)
        
        if profile['total_answers'] > 0:
            profile['learning_potential'] = self._calculate_learning_potential(student_id)
            profile['learning_characteristics'] = self._calculate_learning_characteristics(student_id)
        else:
            profile['learning_potential'] = '未知'
            profile['learning_characteristics'] = {}
        
        return profile
    
    def _calculate_learning_potential(self, student_id: str) -> str:
        """计算学习潜力"""
        history = self.db.get_answer_history(student_id, limit=20)
        
        if len(history) < 5:
            return "中等"
        
        first_half = history[len(history)//2:]
        second_half = history[:len(history)//2]
        
        first_mastery = sum(r['mastery_after'] for r in first_half) / len(first_half)
        second_mastery = sum(r['mastery_after'] for r in second_half) / len(second_half)
        
        improvement = second_mastery - first_mastery
        
        if improvement > 0.1:
            return "高"
        elif improvement > 0.05:
            return "中等"
        else:
            return "需要加强"
    
    def _calculate_learning_characteristics(self, student_id: str) -> Dict[str, Any]:
        """计算学习特征"""
        history = self.db.get_answer_history(student_id, limit=50)
        
        if not history:
            return {'difficulty_preference': '中等', 'learning_stability': 0.0}
        
        difficulty_counts = {'简单': 0, '中等': 0, '困难': 0}
        for r in history:
            diff = r['difficulty']
            if diff < 0.35:
                difficulty_counts['简单'] += 1
            elif diff < 0.65:
                difficulty_counts['中等'] += 1
            else:
                difficulty_counts['困难'] += 1
        
        max_difficulty = max(difficulty_counts, key=difficulty_counts.get)
        
        mastery_values = [r['mastery_after'] for r in history]
        if len(mastery_values) >= 3:
            mean_mastery = sum(mastery_values) / len(mastery_values)
            std_mastery = (sum((x - mean_mastery) ** 2 for x in mastery_values) / len(mastery_values)) ** 0.5
            cv = std_mastery / mean_mastery if mean_mastery > 0 else 1.0
            stability = 1 - min(cv, 1.0)
        else:
            stability = 0.5
        
        return {
            'difficulty_preference': max_difficulty,
            'learning_stability': stability
        }


def create_bkt_database_adapter(db_manager, params: Optional[BKTParameters] = None):
    """创建BKT数据库适配器"""
    return BKTDatabaseAdapter(db_manager, params)