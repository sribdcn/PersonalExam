# -*- coding: utf-8 -*-
"""
系统核心模块 - 数据库版本（知识图谱架构 + 自动检测更新）
"""

import logging
import random
import hashlib
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class SmartEducationSystemDB:
    """智能教育系统核心（数据库版 + 知识图谱 + 自动检测更新）"""
    
    def __init__(self, config, db_manager, bkt_algorithm):
        self.config = config
        self.db = db_manager
        self.bkt_algorithm = bkt_algorithm
        self.embedding_model = None
        self.pangu_model = None
        self.evaluator = None
        self.visualizer = None
        self.knowledge_graph = None
        self.kg_rag = None
        self.question_selector = None
        
        logger.info("✅ 智能教育系统核心初始化（数据库版 + 知识图谱 + 自动检测）")
    
    def _check_database_hash(self) -> str:
        try:
            all_questions = self.db.get_all_questions()
            
            questions_str = json.dumps(
                sorted([q.get('题号', 0) for q in all_questions])
            )
            return hashlib.md5(questions_str.encode()).hexdigest()
        except Exception as e:
            logger.error(f"❌ 计算数据库哈希失败: {e}")
            return ""
    
    def _should_rebuild_kg(self, kg_cache_path: Path, hash_cache_path: Path) -> bool:

        if self.config.SMART_QUESTION_CONFIG.get('rebuild_kg', False):
            logger.info("🔨 配置要求强制重建知识图谱")
            return True

        if not kg_cache_path.exists():
            logger.info("📂 知识图谱缓存不存在，需要构建")
            return True

        current_hash = self._check_database_hash()
        
        if not hash_cache_path.exists():
            logger.info("📂 哈希缓存不存在，假定需要重建")
            return True
        
        try:
            with open(hash_cache_path, 'r') as f:
                cached_hash = f.read().strip()
            
            if current_hash != cached_hash:
                logger.info("🔄 检测到题库变化，需要重建知识图谱")
                logger.info(f"   旧哈希: {cached_hash[:16]}...")
                logger.info(f"   新哈希: {current_hash[:16]}...")
                return True
            else:
                logger.info("✅ 题库未变化，使用缓存的知识图谱")
                return False
                
        except Exception as e:
            logger.warning(f"⚠️  读取哈希缓存失败: {e}，假定需要重建")
            return True
    
    def _save_database_hash(self, hash_cache_path: Path):
        try:
            current_hash = self._check_database_hash()
            hash_cache_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(hash_cache_path, 'w') as f:
                f.write(current_hash)
            
            logger.info(f"💾 数据库哈希已保存: {current_hash[:16]}...")
        except Exception as e:
            logger.error(f"❌ 保存哈希失败: {e}")
    
    def initialize(self):
        logger.info("🔄 初始化系统组件...")
        
        try:
            from models.llm_models import create_llm_model
            from models.embedding_model import create_embedding_model
            from utils.evaluator import create_evaluator
            from visualization.kg_visualizer import create_visualizer
            from knowledge_management.kg_builder import create_kg_builder
            from knowledge_management.rag_engine import create_kg_rag
            from utils.question_generator import create_question_selector
            
            logger.info("🚀 初始化盘古7B模型...")
            self.pangu_model = create_llm_model(
                'pangu',
                self.config.PANGU_MODEL_PATH,
                self.config.PANGU_MODEL_CONFIG
            )
            self.pangu_model.load_model()

            logger.info("🔨 智能构建/加载知识图谱...")
            all_questions = self.db.get_all_questions()
            
            kg_cache_path = self.config.DATA_DIR / "knowledge_graph.pkl"
            hash_cache_path = self.config.DATA_DIR / "kg_hash.txt"
            
            force_rebuild = self._should_rebuild_kg(kg_cache_path, hash_cache_path)
            
            kg_builder = create_kg_builder(self.pangu_model, str(kg_cache_path))
            
            self.knowledge_graph = kg_builder.build_from_questions(
                all_questions,
                force_rebuild=force_rebuild
            )
            
            self._save_database_hash(hash_cache_path)
            
            logger.info(f"✅ 知识图谱就绪: {self.knowledge_graph.number_of_nodes()} 个节点, "
                       f"{self.knowledge_graph.number_of_edges()} 条边")
            
            logger.info("🔤 初始化BGE嵌入模型...")
            self.embedding_model = create_embedding_model(
                self.config.BGE_M3_MODEL_PATH,
                self.config.EMBEDDING_MODEL_CONFIG
            )
            self.embedding_model.load_model()
            
            logger.info("🧠 初始化知识图谱RAG引擎...")
            self.kg_rag = create_kg_rag(self.knowledge_graph, self.embedding_model)
            
            logger.info("📊 初始化评估器...")
            self.evaluator = create_evaluator(
                self.pangu_model,
                self.bkt_algorithm,
                self.config.EVALUATION_CONFIG
            )
            
            logger.info("📝 初始化题目选择器...")
            self.question_selector = create_question_selector(
                self.kg_rag,
                self.db,
                self.config.SMART_QUESTION_CONFIG
            )
            
            logger.info("🎨 初始化可视化...")
            self.visualizer = create_visualizer(self.config.VISUALIZATION_CONFIG)
            self.visualizer.build_graph_from_questions(all_questions)
            
            logger.info("✅ 系统初始化完成（数据库版 + 知识图谱 + 自动检测）")
            
        except Exception as e:
            logger.error(f"❌ 系统初始化失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def start_smart_assessment(self, student_id: str, 
                              num_questions: int = 10) -> Optional[Dict]:
        """开始智能测评"""
        try:
            logger.info(f"🚀 开始测评: 学生 {student_id}, 题数 {num_questions}")
            
            profile = self.bkt_algorithm.generate_student_profile(student_id)
            
            used_ids = set()
            major_point, minor_point = self._select_target_knowledge_point(
                student_id, used_ids
            )
            
            state = self.bkt_algorithm.get_student_state(
                student_id, major_point, minor_point
            )
            student_mastery = state['mastery_prob']
            
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
            
            return session
            
        except Exception as e:
            logger.error(f"❌ 开始测评失败: {e}")
            return None
    
    def submit_answer(self, session: Dict, student_answer: str) -> Dict:
        """提交答案"""
        try:
            question = session['current_question']
            major_point = session['current_major_point']
            minor_point = session['current_minor_point']
            
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
                    student_mastery=updated_state['mastery_prob'],
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
    
    def next_question(self, session: Dict) -> Dict:
        """加载下一题"""
        session['current_index'] += 1
        if session['current_index'] <= len(session['questions']):
            session['current_question'] = session['questions'][session['current_index'] - 1]
        return session
    
    def generate_report(self, session: Dict) -> str:
        """生成报告"""
        try:
            return self.evaluator.generate_comprehensive_report(
                session['student_id'],
                "综合评估",
                session['answer_records']
            )
        except Exception as e:
            logger.error(f"❌ 生成报告失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return f"报告生成失败: {str(e)}"
    
    def _select_target_knowledge_point(self, student_id: str, 
                                       used_question_ids: set) -> Tuple[str, str]:
        """选择目标知识点"""
        weak_points = self.bkt_algorithm.get_weak_knowledge_points(student_id)
        
        if weak_points and random.random() < 0.7:
            for major, minor, _ in weak_points:
                questions = self.db.get_questions_filtered(
                    major_point=major,
                    minor_point=minor
                )
                available = [q for q in questions if q.get('题号') not in used_question_ids]
                if available:
                    return major, minor
        
        all_kp = self.db.get_knowledge_points()
        all_combinations = [(m, n) for m, minors in all_kp.items() for n in minors]
        random.shuffle(all_combinations)
        
        for major, minor in all_combinations:
            questions = self.db.get_questions_filtered(
                major_point=major,
                minor_point=minor
            )
            available = [q for q in questions if q.get('题号') not in used_question_ids]
            if available:
                return major, minor
        
        return all_combinations[0] if all_combinations else ("未知", "未知")
    
    def get_database_statistics(self) -> Dict[str, Any]:
        """获取数据库统计"""
        return self.db.get_question_statistics()
    
    def get_system_info(self) -> str:
        """获取系统信息"""
        return f"""
系统版本: {self.config.SYSTEM_INFO['version']} (数据库版 + 知识图谱 + 自动检测)
模型: {self.config.SYSTEM_INFO['model']}
设备: {self.config.SYSTEM_INFO['device']}

架构特点:
  - 盘古7B构建知识图谱
  - RAG基于知识图谱检索
  - 数据持久化到SQLite
  - 自动检测题库变化并重建知识图谱

知识图谱统计:
  - 节点数: {self.knowledge_graph.number_of_nodes() if self.knowledge_graph else 0}
  - 边数: {self.knowledge_graph.number_of_edges() if self.knowledge_graph else 0}
"""
    
    def force_rebuild_kg(self):
        """手动强制重建知识图谱"""
        try:
            logger.info("🔨 手动触发知识图谱重建...")
            
            from knowledge_management.kg_builder import create_kg_builder
            
            all_questions = self.db.get_all_questions()
            kg_cache_path = self.config.DATA_DIR / "knowledge_graph.pkl"
            hash_cache_path = self.config.DATA_DIR / "kg_hash.txt"
            
            kg_builder = create_kg_builder(self.pangu_model, str(kg_cache_path))
            self.knowledge_graph = kg_builder.build_from_questions(
                all_questions,
                force_rebuild=True
            )
            
            self._save_database_hash(hash_cache_path)
            
            from knowledge_management.rag_engine import create_kg_rag
            self.kg_rag = create_kg_rag(self.knowledge_graph, self.embedding_model)
            
            from utils.question_generator import create_question_selector
            self.question_selector = create_question_selector(
                self.kg_rag,
                self.db,
                self.config.SMART_QUESTION_CONFIG
            )
            
            self.visualizer.build_graph_from_questions(all_questions)
            
            logger.info("✅ 知识图谱重建完成")
            return True
            
        except Exception as e:
            logger.error(f"❌ 知识图谱重建失败: {e}")
            return False


def create_system_core_with_db(config, db_manager, bkt_algorithm):
    """创建系统核心（数据库版）"""
    core = SmartEducationSystemDB(config, db_manager, bkt_algorithm)
    core.initialize()
    return core