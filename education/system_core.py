# -*- coding: utf-8 -*-
"""
系统核心模块 - 基于知识图谱的智能个性化版本

新架构：
1. 盘古7B构建全局知识图谱
2. RAG基于知识图谱检索题目
3. 盘古7B判断答案和生成报告
"""

import logging
import random
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class SmartEducationSystem:
    """智能教育系统核心"""
    
    def __init__(self, config):
        self.config = config
        self.question_db = None
        self.embedding_model = None
        self.pangu_model = None
        self.evaluator = None
        self.visualizer = None
        self.bkt_algorithm = None
        self.knowledge_graph = None  # 知识图谱
        self.kg_rag = None  # 基于KG的RAG
        self.question_selector = None
        self.models_loaded = False
        
        logger.info("✅ 智能教育系统核心初始化（知识图谱版）")
    
    def initialize(self):
        logger.info("🔄 初始化系统组件...")
        
        try:
            from models.llm_models import create_llm_model
            from models.embedding_model import create_embedding_model
            from data_management.question_db import create_question_database
            from utils.evaluator import create_evaluator
            from visualization.kg_visualizer import create_visualizer
            from utils.bkt_algorithm import create_bkt_algorithm
            from knowledge_management.kg_builder import create_kg_builder
            from knowledge_management.rag_engine import create_kg_rag
            from utils.question_generator import create_question_selector

            logger.info("初始化题库...")
            self.question_db = create_question_database(str(self.config.QUESTION_DB))
            all_questions = self.question_db.get_all_questions()
            logger.info(f"✅ 题库加载完成: {len(all_questions)} 道题")

            logger.info("初始化盘古7B模型...")
            self.pangu_model = create_llm_model(
                'pangu',
                self.config.PANGU_MODEL_PATH,
                self.config.PANGU_MODEL_CONFIG
            )
            self.pangu_model.load_model()
            logger.info("盘古7B模型加载完成")

            logger.info("使用盘古7B构建知识图谱...")
            kg_cache_path = self.config.DATA_DIR / "knowledge_graph.pkl"
            kg_builder = create_kg_builder(self.pangu_model, str(kg_cache_path))

            force_rebuild = self.config.SMART_QUESTION_CONFIG.get('rebuild_kg', False)
            self.knowledge_graph = kg_builder.build_from_questions(
                all_questions,
                force_rebuild=force_rebuild
            )
            logger.info(f"✅ 知识图谱构建完成: {self.knowledge_graph.number_of_nodes()} 个节点")

            logger.info("🔤 初始化BGE嵌入模型...")
            self.embedding_model = create_embedding_model(
                self.config.BGE_M3_MODEL_PATH,
                self.config.EMBEDDING_MODEL_CONFIG
            )
            self.embedding_model.load_model()
            logger.info("✅ BGE嵌入模型加载完成")

            logger.info("🧠 初始化知识图谱RAG引擎...")
            self.kg_rag = create_kg_rag(self.knowledge_graph, self.embedding_model)
            logger.info("✅ 知识图谱RAG引擎初始化完成")

            logger.info("🧠 初始化BKT算法...")
            self.bkt_algorithm = create_bkt_algorithm(
                storage_path=str(self.config.DATA_DIR / "student_states.json")
            )
            
            logger.info("📊 初始化评估器...")
            self.evaluator = create_evaluator(
                self.pangu_model,
                self.bkt_algorithm,
                self.config.EVALUATION_CONFIG
            )

            logger.info("📝 初始化题目选择器...")
            self.question_selector = create_question_selector(
                self.kg_rag,
                self.question_db,
                self.config.SMART_QUESTION_CONFIG
            )
            
            logger.info("🎨 初始化可视化组件...")
            self.visualizer = create_visualizer(self.config.VISUALIZATION_CONFIG)
            self.visualizer.build_graph_from_questions(all_questions)
            
            self.models_loaded = True
            logger.info("✅ 系统初始化完成 - 知识图谱智能学习版")
            
        except Exception as e:
            logger.error(f"❌ 系统初始化失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise RuntimeError(f"系统初始化失败: {e}")
    
    def start_smart_assessment(self, student_id: str = "default_student",
                              num_questions: int = 10) -> Optional[Dict[str, Any]]:
        """开始智能测评"""
        try:
            logger.info(f"🚀 开始智能测评: 学生 {student_id}, 题数 {num_questions}")
            

            profile = self.bkt_algorithm.generate_student_profile(student_id)
            

            used_ids = set()
            major_point, minor_point = self._select_target_knowledge_point(
                student_id, used_ids
            )
            

            state = self.bkt_algorithm.get_student_state(student_id, major_point, minor_point)
            student_mastery = state.mastery_prob
            

            logger.info(f"🔍 使用知识图谱RAG检索题目...")
            first_question = self.question_selector.select_question(
                student_id=student_id,
                student_mastery=student_mastery,
                major_point=major_point,
                minor_point=minor_point,
                used_question_ids=used_ids
            )
            
            if not first_question:
                logger.error("❌ 无法选择第一题")
                return None
            
            used_ids.add(first_question.get('题号'))
            
            session = {
                'student_id': student_id,
                'total_questions': num_questions,
                'current_index': 1,
                'current_question': first_question,
                'current_major_point': major_point,
                'current_minor_point': minor_point,
                'questions': [first_question],
                'answer_records': [],
                'last_result': None,
                'used_question_ids': used_ids,
                'profile': profile
            }
            
            logger.info(f"测评开始 - 第1题: {major_point}/{minor_point}")
            return session
            
        except Exception as e:
            logger.error(f"开始测评失败: {e}")
            return None
    
    def submit_answer(self, session: Dict[str, Any], 
                     student_answer: str) -> Dict[str, Any]:
        try:
            question = session['current_question']
            major_point = session['current_major_point']
            minor_point = session['current_minor_point']
            
            logger.info(f"🤖 使用盘古7B评估答案...")
            
            is_correct, reason = self.evaluator.check_answer(
                question,
                student_answer,
                self.config.PROMPTS['answer_check']
            )
            

            bkt_result = self.bkt_algorithm.record_answer(
                session['student_id'],
                major_point,
                minor_point,
                question,
                is_correct
            )
            
            record = {
                'question': question,
                'major_point': major_point,
                'minor_point': minor_point,
                'student_answer': student_answer,
                'is_correct': is_correct,
                'check_reason': reason,
                'mastery_before': bkt_result['previous_mastery'],
                'mastery_after': bkt_result['current_mastery'],
                'mastery_change': bkt_result['mastery_change']
            }
            
            session['answer_records'].append(record)
            session['last_result'] = record
            
            if session['current_index'] < session['total_questions']:
                next_major, next_minor = self._select_target_knowledge_point(
                    session['student_id'],
                    session['used_question_ids']
                )
                
                updated_state = self.bkt_algorithm.get_student_state(
                    session['student_id'],
                    next_major,
                    next_minor
                )
                
                next_question = self.question_selector.select_question(
                    student_id=session['student_id'],
                    student_mastery=updated_state.mastery_prob,
                    major_point=next_major,
                    minor_point=next_minor,
                    used_question_ids=session['used_question_ids']
                )
                
                if next_question:
                    session['questions'].append(next_question)
                    session['used_question_ids'].add(next_question.get('题号'))
                    session['current_major_point'] = next_major
                    session['current_minor_point'] = next_minor
            
            return session
            
        except Exception as e:
            logger.error(f"❌ 提交答案失败: {e}")
            return session
    
    def next_question(self, session: Dict[str, Any]) -> Dict[str, Any]:
        """加载下一题"""
        session['current_index'] += 1
        
        if session['current_index'] <= len(session['questions']):
            session['current_question'] = session['questions'][session['current_index'] - 1]
        
        return session
    
    def generate_report(self, session: Dict[str, Any]) -> str:
        """生成报告（盘古7B）"""
        try:
            logger.info("📝 使用盘古7B生成评估报告...")
            report = self.evaluator.generate_comprehensive_report(
                session['student_id'],
                "综合评估",
                session['answer_records']
            )
            return report
        except Exception as e:
            logger.error(f"❌ 生成报告失败: {e}")
            return f"报告生成失败: {str(e)}"
    
    def _select_target_knowledge_point(self, student_id: str, 
                                       used_question_ids: set) -> Tuple[str, str]:
        """选择目标知识点"""
        weak_points = self.bkt_algorithm.get_weak_knowledge_points(student_id)
        
        if weak_points and random.random() < 0.7:
            for major, minor, _ in weak_points:
                questions = self.question_db.get_questions_filtered(
                    major_point=major,
                    minor_point=minor
                )
                available = [q for q in questions if q.get('题号') not in used_question_ids]
                if available:
                    return major, minor
        
        all_kp = self.question_db.get_all_knowledge_points()
        all_combinations = [(m, n) for m, minors in all_kp.items() for n in minors]
        random.shuffle(all_combinations)
        
        for major, minor in all_combinations:
            questions = self.question_db.get_questions_filtered(
                major_point=major,
                minor_point=minor
            )
            available = [q for q in questions if q.get('题号') not in used_question_ids]
            if available:
                return major, minor
        
        return all_combinations[0] if all_combinations else ("未知", "未知")
    
    def get_system_info(self) -> str:
        """获取系统信息"""
        student_count = len(self.bkt_algorithm.student_states) if self.bkt_algorithm else 0
        
        info = f"""
系统版本: {self.config.SYSTEM_INFO['version']}
架构: 知识图谱 + RAG + 盘古7B

核心流程:
  1. 系统启动 → 盘古7B构建知识图谱
  2. 选题时 → RAG基于知识图谱检索
  3. 答题后 → 盘古7B判断正误
  4. 测评结束 → 盘古7B生成报告

技术栈:
  - 语言模型: 盘古7B (构建KG、判断答案、生成报告)
  - 知识图谱: NetworkX (节点 {self.knowledge_graph.number_of_nodes()}, 边 {self.knowledge_graph.number_of_edges()})
  - RAG引擎: 基于知识图谱的智能检索
  - 嵌入模型: BGE-small-zh (辅助RAG)
  - 学习建模: BKT算法 (贝叶斯知识追踪)

数据统计:
  - 题库: {len(self.question_db.get_all_questions())} 道题
  - 学生: {student_count} 人
  - 设备: {self.config.SYSTEM_INFO['device']}
"""
        return info
    
    def get_database_statistics(self) -> Dict[str, Any]:
        """获取数据库统计"""
        return self.question_db.get_statistics()
    
    def reload_models(self):
        """重新加载模型"""
        if self.pangu_model:
            self.pangu_model.load_model()
    
    def clear_cache(self):
        """清除缓存"""
        import torch
        try:
            import torch_npu
            if torch.npu.is_available():
                for i in range(torch.npu.device_count()):
                    torch.npu.empty_cache()
        except:
            pass


def create_system_core(config):
    """创建系统核心"""
    core = SmartEducationSystem(config)
    core.initialize()
    return core