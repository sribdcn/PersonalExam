# -*- coding: utf-8 -*-
"""
Copyright (c) 2025 AI系统与应用课题组@SRIBD

基于LLM和知识图谱协同的个性化出题系统 (PersonalExam)
Personalized Question Generation System Based on LLM and Knowledge Graph Collaboration

贝叶斯知识追踪(BKT)算法模块 - 细粒度知识点版本
"""

import logging
import numpy as np
import json
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class BKTParameters:
    """BKT算法参数"""
    p_init: float = 0.3      # 初始掌握概率
    p_learn: float = 0.2     # 学习概率
    p_guess: float = 0.3     # 猜测概率
    p_slip: float = 0.1      # 失误概率
    p_forget: float = 0.05   # 遗忘概率


@dataclass
class StudentState:
    """学生状态记录"""
    student_id: str
    knowledge_point_major: str      # 知识点大类
    knowledge_point_minor: str      # 知识点小类
    mastery_prob: float             # 当前掌握概率
    answer_history: List[Dict[str, Any]]  # 历史答题记录
    recent_performance: List[bool]  # 最近表现
    params: BKTParameters           # 参数
    created_at: str = ""
    updated_at: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()


class EnhancedBKT:
    """增强版贝叶斯知识追踪"""
    
    def __init__(self, default_params: Optional[BKTParameters] = None,
                 storage_path: str = "./data/student_states.json"):
        self.default_params = default_params or BKTParameters()
        self.storage_path = Path(storage_path)
        # 三层结构：student_id -> major_point -> minor_point -> state
        self.student_states: Dict[str, Dict[str, Dict[str, StudentState]]] = defaultdict(lambda: defaultdict(dict))
        
        self._load_states()
        logger.info(f"✅ 增强版BKT算法初始化完成")
    
    def _load_states(self):
        """加载学生状态"""
        if not self.storage_path.exists():
            logger.info("📂 学生状态文件不存在，创建新文件")
            return
        
        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for student_id, major_points in data.items():
                for major_point, minor_points in major_points.items():
                    for minor_point, state_dict in minor_points.items():
                        params = BKTParameters(**state_dict.get('params', {}))
                        state = StudentState(
                            student_id=state_dict['student_id'],
                            knowledge_point_major=state_dict['knowledge_point_major'],
                            knowledge_point_minor=state_dict['knowledge_point_minor'],
                            mastery_prob=state_dict['mastery_prob'],
                            answer_history=state_dict['answer_history'],
                            recent_performance=state_dict['recent_performance'],
                            params=params,
                            created_at=state_dict.get('created_at', ''),
                            updated_at=state_dict.get('updated_at', '')
                        )
                        self.student_states[student_id][major_point][minor_point] = state
            
            total_students = len(self.student_states)
            total_records = sum(
                sum(len(minor) for minor in major.values()) 
                for major in self.student_states.values()
            )
            logger.info(f"✅ 加载学生状态: {total_students} 个学生, {total_records} 条细粒度记录")
            
        except Exception as e:
            logger.error(f"❌ 加载学生状态失败: {e}")
            self.student_states = defaultdict(lambda: defaultdict(dict))
    
    def _save_states(self):
        """保存学生状态"""
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {}
            for student_id, major_points in self.student_states.items():
                data[student_id] = {}
                for major_point, minor_points in major_points.items():
                    data[student_id][major_point] = {}
                    for minor_point, state in minor_points.items():
                        data[student_id][major_point][minor_point] = {
                            'student_id': state.student_id,
                            'knowledge_point_major': state.knowledge_point_major,
                            'knowledge_point_minor': state.knowledge_point_minor,
                            'mastery_prob': state.mastery_prob,
                            'answer_history': state.answer_history,
                            'recent_performance': state.recent_performance,
                            'params': asdict(state.params),
                            'created_at': state.created_at,
                            'updated_at': state.updated_at
                        }
            
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.debug(f"💾 学生状态已保存")
            
        except Exception as e:
            logger.error(f"❌ 保存学生状态失败: {e}")
    
    def get_student_state(self, student_id: str, major_point: str, 
                         minor_point: str) -> StudentState:
        """获取或初始化学生状态"""
        if (student_id not in self.student_states or 
            major_point not in self.student_states[student_id] or
            minor_point not in self.student_states[student_id][major_point]):
            
            params = self._get_personalized_params(student_id)
            state = StudentState(
                student_id=student_id,
                knowledge_point_major=major_point,
                knowledge_point_minor=minor_point,
                mastery_prob=params.p_init,
                answer_history=[],
                recent_performance=[],
                params=params
            )
            
            self.student_states[student_id][major_point][minor_point] = state
            self._save_states()
            
            logger.info(f"🆕 初始化学生状态: {student_id} - {major_point}/{minor_point}")
        
        return self.student_states[student_id][major_point][minor_point]
    
    def _get_personalized_params(self, student_id: str) -> BKTParameters:
        """获取个性化参数"""
        if student_id not in self.student_states:
            return self.default_params
        
        all_history = []
        for major_points in self.student_states[student_id].values():
            for state in major_points.values():
                all_history.extend(state.answer_history)
        
        if len(all_history) < 10:
            return self.default_params
        
        total = len(all_history)
        correct = sum(1 for r in all_history if r.get('is_correct', False))
        accuracy = correct / total
        
        params = BKTParameters()
        
        if accuracy > 0.8:
            params.p_init = 0.5
        elif accuracy > 0.6:
            params.p_init = 0.4
        else:
            params.p_init = 0.2
        
        learning_speed = self._calculate_learning_speed(all_history)
        if learning_speed > 0.1:
            params.p_learn = 0.3
        elif learning_speed > 0.05:
            params.p_learn = 0.2
        else:
            params.p_learn = 0.15
        
        return params
    
    def _calculate_learning_speed(self, history: List[Dict[str, Any]]) -> float:
        """计算学习速度"""
        if len(history) < 3:
            return 0.0
        
        changes = []
        for i in range(1, len(history)):
            prev = history[i-1].get('mastery_after', 0.3)
            curr = history[i].get('mastery_after', 0.3)
            changes.append(curr - prev)
        
        return sum(changes) / len(changes) if changes else 0.0
    
    def update_mastery_probability(self, state: StudentState, is_correct: bool) -> float:
        """更新掌握概率"""
        p_mastery = state.mastery_prob
        p_learn = state.params.p_learn
        p_forget = state.params.p_forget
        p_guess = state.params.p_guess
        p_slip = state.params.p_slip
        
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
        state.mastery_prob = p_mastery_updated
        state.updated_at = datetime.now().isoformat()
        
        return p_mastery_updated
    
    def record_answer(self, student_id: str, major_point: str, minor_point: str,
                     question: Dict[str, Any], is_correct: bool) -> Dict[str, Any]:
        """记录答题并更新状态"""
        state = self.get_student_state(student_id, major_point, minor_point)
        
        answer_record = {
            'question': question,
            'is_correct': is_correct,
            'timestamp': datetime.now().isoformat(),
            'difficulty': question.get('难度', 0.5),
            'mastery_before': state.mastery_prob
        }
        
        new_mastery = self.update_mastery_probability(state, is_correct)
        answer_record['mastery_after'] = new_mastery
        
        state.recent_performance.append(is_correct)
        if len(state.recent_performance) > 10:
            state.recent_performance.pop(0)
        
        state.answer_history.append(answer_record)
        self._save_states()
        
        result = {
            'student_id': student_id,
            'knowledge_point_major': major_point,
            'knowledge_point_minor': minor_point,
            'current_mastery': new_mastery,
            'previous_mastery': answer_record['mastery_before'],
            'mastery_change': new_mastery - answer_record['mastery_before'],
            'total_answers': len(state.answer_history),
            'recent_accuracy': self._calculate_recent_accuracy(state)
        }
        
        logger.info(f"📝 答题记录: {student_id} - {minor_point}: "
                   f"{answer_record['mastery_before']:.3f} → {new_mastery:.3f} "
                   f"({'✓' if is_correct else '✗'})")
        
        return result
    
    def _calculate_recent_accuracy(self, state: StudentState) -> float:
        """计算最近准确率"""
        if not state.recent_performance:
            return 0.0
        return sum(state.recent_performance) / len(state.recent_performance)
    
    def get_weak_knowledge_points(self, student_id: str, 
                                  threshold: float = 0.4) -> List[Tuple[str, str, float]]:
        """
        获取学生薄弱的知识点小类
        
        Returns:
            List of (major_point, minor_point, mastery_prob)
        """
        weak_points = []
        
        if student_id in self.student_states:
            for major_point, minor_points in self.student_states[student_id].items():
                for minor_point, state in minor_points.items():
                    if state.mastery_prob < threshold:
                        weak_points.append((major_point, minor_point, state.mastery_prob))
        
        # 按掌握度升序排序（最薄弱的在前）
        weak_points.sort(key=lambda x: x[2])
        
        return weak_points
    
    def get_all_mastery_status(self, student_id: str) -> Dict[str, Dict[str, float]]:
        """获取学生所有知识点的掌握情况"""
        status = defaultdict(dict)
        
        if student_id in self.student_states:
            for major_point, minor_points in self.student_states[student_id].items():
                for minor_point, state in minor_points.items():
                    status[major_point][minor_point] = state.mastery_prob
        
        return dict(status)
    
    def generate_student_profile(self, student_id: str) -> Dict[str, Any]:
        """生成学生评估画像"""
        if student_id not in self.student_states:
            return {
                'student_id': student_id,
                'knowledge_points': {},
                'overall_mastery': 0.0,
                'weak_points': [],
                'strengths': [],
                'total_knowledge_points': 0,  # 添加这个字段
                'total_answers': 0,           # 添加这个字段
                'learning_potential': '未知',  # 添加这个字段
                'learning_characteristics': {} # 添加这个字段
            }
        
        all_states = []
        knowledge_points = defaultdict(dict)
        
        for major_point, minor_points in self.student_states[student_id].items():
            for minor_point, state in minor_points.items():
                all_states.append(state)
                knowledge_points[major_point][minor_point] = {
                    'mastery': state.mastery_prob,
                    'total_answers': len(state.answer_history),
                    'recent_accuracy': self._calculate_recent_accuracy(state),
                    'updated_at': state.updated_at
                }
        
        overall_mastery = sum(s.mastery_prob for s in all_states) / len(all_states) if all_states else 0.0
        
        weak_points = self.get_weak_knowledge_points(student_id, threshold=0.4)
        strengths = [(maj, min, s.mastery_prob) 
                    for maj, minors in self.student_states[student_id].items()
                    for min, s in minors.items()
                    if s.mastery_prob > 0.7]
        
        # 计算学习潜力（基于进步速度和稳定性）
        learning_potential = self._calculate_learning_potential(all_states)
        
        # 计算学习特征
        learning_characteristics = self._calculate_learning_characteristics(all_states)
        
        return {
            'student_id': student_id,
            'knowledge_points': dict(knowledge_points),
            'overall_mastery': overall_mastery,
            'weak_points': weak_points[:5],  # 前5个最薄弱的
            'strengths': strengths,
            'total_knowledge_points': len(all_states),
            'total_answers': sum(len(s.answer_history) for s in all_states),
            'learning_potential': learning_potential,  # 添加学习潜力
            'learning_characteristics': learning_characteristics  # 添加学习特征
        }

    def _calculate_learning_potential(self, all_states: List[StudentState]) -> str:
        """计算学习潜力"""
        if not all_states:
            return "未知"
        
        # 基于平均学习速度
        learning_speeds = []
        for state in all_states:
            if len(state.answer_history) >= 2:
                changes = []
                for i in range(1, len(state.answer_history)):
                    prev = state.answer_history[i-1].get('mastery_after', 0.3)
                    curr = state.answer_history[i].get('mastery_after', 0.3)
                    changes.append(curr - prev)
                if changes:
                    learning_speeds.append(sum(changes) / len(changes))
        
        if not learning_speeds:
            return "中等"
        
        avg_speed = sum(learning_speeds) / len(learning_speeds)
        
        if avg_speed > 0.1:
            return "高"
        elif avg_speed > 0.05:
            return "中等"
        else:
            return "需要加强"

    def _calculate_learning_characteristics(self, all_states: List[StudentState]) -> Dict[str, Any]:
        """计算学习特征"""
        if not all_states:
            return {
                'difficulty_preference': '中等',
                'learning_stability': 0.0
            }
        
        # 分析难度偏好
        difficulty_counts = {'简单': 0, '中等': 0, '困难': 0}
        for state in all_states:
            for record in state.answer_history:
                diff = record.get('question', {}).get('难度', 0.5)
                if diff < 0.35:
                    difficulty_counts['简单'] += 1
                elif diff < 0.65:
                    difficulty_counts['中等'] += 1
                else:
                    difficulty_counts['困难'] += 1
        
        # 找出最常做的难度
        max_difficulty = max(difficulty_counts, key=difficulty_counts.get)
        
        # 计算学习稳定性（基于掌握度波动）
        stability_scores = []
        for state in all_states:
            if len(state.answer_history) >= 3:
                mastery_values = [r.get('mastery_after', 0.3) for r in state.answer_history]
                if len(mastery_values) >= 3:
                    # 计算变异系数
                    mean_mastery = sum(mastery_values) / len(mastery_values)
                    std_mastery = (sum((x - mean_mastery) ** 2 for x in mastery_values) / len(mastery_values)) ** 0.5
                    cv = std_mastery / mean_mastery if mean_mastery > 0 else 1.0
                    stability_scores.append(1 - min(cv, 1.0))
        
        avg_stability = sum(stability_scores) / len(stability_scores) if stability_scores else 0.5
        
        return {
            'difficulty_preference': max_difficulty,
            'learning_stability': avg_stability
        }


def create_bkt_algorithm(params: Optional[BKTParameters] = None,
                        storage_path: str = "./data/student_states.json") -> EnhancedBKT:
    """创建增强版BKT算法实例"""
    return EnhancedBKT(params, storage_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    bkt = create_bkt_algorithm()
    
    # 测试
    test_question = {
        '问题': 'x^2 - 5x + 6 = 0',
        '答案': 'x = 2 或 x = 3',
        '难度': 0.3
    }
    
    result = bkt.record_answer("test_student", "代数", "一元二次方程", test_question, True)
    print(f"掌握度: {result['previous_mastery']:.3f} → {result['current_mastery']:.3f}")
    
    weak = bkt.get_weak_knowledge_points("test_student")
    print(f"薄弱知识点: {weak}")