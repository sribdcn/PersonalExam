# -*- coding: utf-8 -*-
"""
Copyright (c) 2025 AI系统及应用课题组@SRIBD

基于LLM和知识图谱协同的个性化出题系统 (PersonalExam)
Personalized Question Generation System Based on LLM and Knowledge Graph Collaboration

系统核心模块 - 基于本地RAG的智能个性化版本（优化知识点选择策略）
使用本地嵌入模型和盘古7B
"""

import logging
import random
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class SmartEducationSystem:
    """基于LLM和知识图谱协同的个性化出题系统核心（本地RAG版）"""
    
    def __init__(self, config):
        self.config = config
        self.question_db = None
        self.embedding_model = None
        self.pangu_model = None
        self.evaluator = None
        self.visualizer = None
        self.bkt_algorithm = None
        self.rag_engine = None
        self.question_selector = None
        self.models_loaded = False
        
        logger.info("✅ 智能教育系统核心初始化（本地RAG版）")
    
    def initialize(self):
        """初始化系统组件"""
        logger.info("🔄 初始化系统组件...")
        
        try:
            from models.llm_models import create_llm_model
            from models.embedding_model import create_embedding_model
            from data_management.question_db import create_question_database
            from utils.evaluator import create_evaluator
            from visualization.kg_visualizer import create_visualizer
            from utils.bkt_algorithm import create_bkt_algorithm
            from knowledge_management.rag_engine import create_rag_engine
            from utils.question_generator import create_question_selector
            
            # 1. 初始化题库
            logger.info("📚 初始化题库...")
            self.question_db = create_question_database(str(self.config.QUESTION_DB))
            
            # 2. 初始化嵌入模型（BGE）
            logger.info("🔤 初始化BGE嵌入模型...")
            self.embedding_model = create_embedding_model(
                self.config.BGE_M3_MODEL_PATH,
                self.config.EMBEDDING_MODEL_CONFIG
            )
            self.embedding_model.load_model()
            logger.info("✅ BGE嵌入模型加载完成")
            
            # 3. 初始化盘古7B模型
            logger.info("🚀 初始化盘古7B模型...")
            self.pangu_model = create_llm_model(
                'pangu',
                self.config.PANGU_MODEL_PATH,
                self.config.PANGU_MODEL_CONFIG
            )
            
            logger.info("🔄 预加载盘古7B模型...")
            self.pangu_model.load_model()
            logger.info("✅ 盘古7B模型加载完成")
            
            # 4. 初始化RAG引擎（本地，不使用LightRAG）
            logger.info("🧠 初始化本地RAG引擎...")
            self.rag_engine = create_rag_engine(
                self.embedding_model,
                self.pangu_model
            )
            
            # 5. 构建题目索引
            logger.info("🔄 构建题目向量索引...")
            all_questions = self.question_db.get_all_questions()
            self.rag_engine.build_question_index(all_questions)
            logger.info("✅ 题目索引构建完成")
            
            # 6. 初始化BKT算法
            logger.info("🧠 初始化BKT算法...")
            self.bkt_algorithm = create_bkt_algorithm(
                storage_path=str(self.config.DATA_DIR / "student_states.json")
            )
            
            # 7. 初始化评估器（使用盘古7B）
            logger.info("📊 初始化评估器（盘古7B驱动）...")
            self.evaluator = create_evaluator(
                self.pangu_model,
                self.bkt_algorithm,
                self.config.EVALUATION_CONFIG
            )
            
            # 8. 初始化题目选择器（RAG + 盘古7B）
            logger.info("📝 初始化智能题目选择器（RAG + 盘古7B）...")
            self.question_selector = create_question_selector(
                self.rag_engine,
                self.pangu_model,
                self.question_db
            )
            
            # 9. 初始化可视化
            logger.info("🎨 初始化可视化组件...")
            self.visualizer = create_visualizer(
                self.config.VISUALIZATION_CONFIG
            )
            
            # 构建知识图谱（从题库）
            logger.info("🔄 正在构建知识图谱...")
            questions = self.question_db.get_all_questions()
            self.visualizer.build_graph_from_questions(questions)
            logger.info("✅ 知识图谱构建完成")
            
            self.models_loaded = True
            logger.info("✅ 系统初始化完成 - 本地RAG智能个性化学习版")
            
        except Exception as e:
            logger.error(f"❌ 系统初始化失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise RuntimeError(f"系统初始化失败: {e}")
    
    def _analyze_student_weakness(self, student_id: str) -> List[Tuple[str, str, float]]:
        """
        分析学生薄弱知识点
        
        Returns:
            List of (major_point, minor_point, mastery) tuples
        """
        weak_threshold = self.config.SMART_QUESTION_CONFIG['weak_threshold']
        weak_points = self.bkt_algorithm.get_weak_knowledge_points(
            student_id, threshold=weak_threshold
        )
        
        if weak_points:
            logger.info(f"📊 识别到 {len(weak_points)} 个薄弱知识点:")
            for major, minor, mastery in weak_points[:3]:
                logger.info(f"  - {major}/{minor}: {mastery:.3f}")
        else:
            logger.info(f"📊 学生 {student_id} 无历史数据或无明显薄弱点")
        
        return weak_points
    
    def _get_unexplored_points(self, student_id: str) -> List[Tuple[str, str]]:
        """
        获取学生未探索的知识点
        
        Returns:
            List of (major_point, minor_point)
        """
        all_knowledge_points = self.question_db.get_all_knowledge_points()
        mastered_status = self.bkt_algorithm.get_all_mastery_status(student_id)
        
        unexplored = []
        for major, minors in all_knowledge_points.items():
            for minor in minors:
                if major not in mastered_status or minor not in mastered_status[major]:
                    unexplored.append((major, minor))
        
        return unexplored
    
    def _check_knowledge_point_has_questions(self, major_point: str, 
                                            minor_point: str,
                                            used_question_ids: set) -> bool:
        """
        检查某个知识点小类是否还有未使用的题目
        
        Args:
            major_point: 知识点大类
            minor_point: 知识点小类
            used_question_ids: 已使用的题目ID
            
        Returns:
            是否有可用题目
        """
        questions = self.question_db.get_questions_filtered(
            major_point=major_point,
            minor_point=minor_point
        )
        available = [q for q in questions if q.get('题号') not in used_question_ids]
        return len(available) > 0
    
    def _select_target_knowledge_point(self, student_id: str, 
                                       used_question_ids: set,
                                       weak_point_ratio: float = 0.7) -> Tuple[str, str]:
        """
        智能选择目标知识点（增强版 - 检查题目可用性）
        
        Args:
            student_id: 学生ID
            used_question_ids: 已使用的题目ID集合
            weak_point_ratio: 选择薄弱点的概率
            
        Returns:
            (major_point, minor_point)
        """
        # 获取薄弱知识点（带掌握度）
        weak_points = self._analyze_student_weakness(student_id)
        
        # 获取未探索知识点
        unexplored_points = self._get_unexplored_points(student_id)
        
        # 策略1：优先加强薄弱点（如果有可用题目）
        if weak_points and random.random() < weak_point_ratio:
            logger.info("🎯 策略：优先加强薄弱知识点")
            # 遍历薄弱点，找到有可用题目的
            for major, minor, mastery in weak_points:
                if self._check_knowledge_point_has_questions(major, minor, used_question_ids):
                    logger.info(f"✅ 选择薄弱知识点: {major}/{minor} (掌握度: {mastery:.3f})")
                    return major, minor
                else:
                    logger.debug(f"⚠️  薄弱知识点 {major}/{minor} 无可用题目，尝试下一个")
            
            # 如果所有薄弱知识点小类都没题了，尝试选择同一大类下的其他小类
            logger.info("⚠️  所有薄弱知识点小类都无可用题目，尝试同大类其他小类")
            weak_major_points = list(set([major for major, _, _ in weak_points]))
            for major in weak_major_points:
                # 获取该大类下所有小类
                all_minors = self.question_db.get_all_knowledge_points().get(major, [])
                # 随机尝试其他小类
                random.shuffle(all_minors)
                for minor in all_minors:
                    if self._check_knowledge_point_has_questions(major, minor, used_question_ids):
                        logger.info(f"✅ 同大类备选: {major}/{minor}")
                        return major, minor
        
        # 策略2：探索新知识点（如果有可用题目）
        if unexplored_points:
            logger.info("🔍 策略：探索新知识点")
            random.shuffle(unexplored_points)
            for major, minor in unexplored_points:
                if self._check_knowledge_point_has_questions(major, minor, used_question_ids):
                    logger.info(f"✅ 选择未探索知识点: {major}/{minor}")
                    return major, minor
                else:
                    logger.debug(f"⚠️  未探索知识点 {major}/{minor} 无可用题目")
        
        # 策略3：随机选择任意有题目的知识点
        logger.warning("⚠️  薄弱点和未探索点均无可用题目，随机选择")
        all_kp = self.question_db.get_all_knowledge_points()
        all_combinations = []
        for major, minors in all_kp.items():
            for minor in minors:
                all_combinations.append((major, minor))
        
        random.shuffle(all_combinations)
        for major, minor in all_combinations:
            if self._check_knowledge_point_has_questions(major, minor, used_question_ids):
                logger.info(f"✅ 随机选择有题知识点: {major}/{minor}")
                return major, minor
        
        # 策略4：实在没办法了，随机返回一个（即使没题目）
        logger.error("❌ 所有知识点都无可用题目！返回随机知识点")
        if all_combinations:
            selected = random.choice(all_combinations)
            logger.error(f"⚠️  强制选择: {selected[0]}/{selected[1]} (可能无题)")
            return selected
        else:
            # 连知识点都没有了
            logger.critical("❌ 题库中没有任何知识点定义！")
            return "未知", "未知"
    
    def start_smart_assessment(self, student_id: str = "default_student",
                              num_questions: int = 10) -> Optional[Dict[str, Any]]:
        """
        开始智能测评（基于RAG的自适应测评）
        
        Args:
            student_id: 学生ID
            num_questions: 题目数量
            
        Returns:
            会话状态
        """
        try:
            logger.info(f"🚀 开始智能测评: 学生 {student_id}, 题数 {num_questions}")
            logger.info(f"📊 使用RAG + BKT算法进行自适应题目选择...")
            
            # 分析学生情况
            profile = self.bkt_algorithm.generate_student_profile(student_id)
            
            total_kp = profile.get('total_knowledge_points', 0)
            overall_mastery = profile.get('overall_mastery', 0.0)
            
            logger.info(f"📊 学生档案: 整体掌握度 {overall_mastery:.3f}, "
                       f"已学知识点 {total_kp}")
            
            # 初始化已使用题目集合
            used_ids = set()
            
            # 选择第一个目标知识点（智能推荐，检查可用性）
            major_point, minor_point = self._select_target_knowledge_point(
                student_id, used_ids
            )
            
            # 获取该知识点的掌握度
            state = self.bkt_algorithm.get_student_state(student_id, major_point, minor_point)
            student_mastery = state.mastery_prob
            
            # 使用RAG选择第一题
            logger.info(f"🔍 使用RAG + 多级备用策略检索题目...")
            first_question = self.question_selector.select_question(
                student_id=student_id,
                student_mastery=student_mastery,
                major_point=major_point,
                minor_point=minor_point,
                used_question_ids=used_ids
            )
            
            if not first_question:
                logger.error("❌ 所有策略均未能选择第一题，题库可能已全部使用完毕")
                return None
            
            used_ids.add(first_question.get('题号'))
            
            # 创建会话
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
            
            logger.info(f"✅ 测评开始 - 第1题: {major_point}/{minor_point}, "
                       f"题号{first_question.get('题号')}")
            return session
            
        except Exception as e:
            logger.error(f"❌ 开始测评失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def submit_answer(self, session: Dict[str, Any], 
                     student_answer: str) -> Dict[str, Any]:
        """
        提交答案（使用盘古7B评估）
        """
        try:
            question = session['current_question']
            major_point = session['current_major_point']
            minor_point = session['current_minor_point']
            
            logger.info(f"✍️  评估答案 (题目 {session['current_index']}/{session['total_questions']})")
            logger.info(f"🤖 使用盘古7B进行严格答案评估...")
            
            # 使用盘古7B检查答案
            is_correct, reason = self.evaluator.check_answer(
                question,
                student_answer,
                self.config.PROMPTS['answer_check']
            )
            
            logger.info(f"📊 盘古7B判定: {'✅ 正确' if is_correct else '❌ 错误'}")
            
            # 记录到BKT（更新掌握度）
            bkt_result = self.bkt_algorithm.record_answer(
                session['student_id'],
                major_point,
                minor_point,
                question,
                is_correct
            )
            
            # 记录答题
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
            
            # 如果还有后续题目，使用RAG选择下一题
            if session['current_index'] < session['total_questions']:
                logger.info(f"🤔 基于RAG + BKT + 多级备用策略智能选择下一题...")
                
                # 选择下一个目标知识点（智能，带可用性检查）
                next_major, next_minor = self._select_target_knowledge_point(
                    session['student_id'],
                    session['used_question_ids']
                )
                
                # 获取更新后的掌握度
                updated_state = self.bkt_algorithm.get_student_state(
                    session['student_id'],
                    next_major,
                    next_minor
                )
                updated_mastery = updated_state.mastery_prob
                
                # 使用RAG + 多级备用策略选择题目
                next_question = self.question_selector.select_question(
                    student_id=session['student_id'],
                    student_mastery=updated_mastery,
                    major_point=next_major,
                    minor_point=next_minor,
                    used_question_ids=session['used_question_ids']
                )
                
                if next_question:
                    session['questions'].append(next_question)
                    session['used_question_ids'].add(next_question.get('题号'))
                    session['current_major_point'] = next_major
                    session['current_minor_point'] = next_minor
                    logger.info(f"✅ 准备下一题: {next_major}/{next_minor}, "
                               f"题号{next_question.get('题号')}")
                else:
                    logger.warning("⚠️  所有策略均无法选择下一题，提前结束测评")
                    logger.warning(f"📊 统计：题库总题目{len(self.question_db.get_all_questions())}道，"
                                 f"已使用{len(session['used_question_ids'])}道")
                    session['total_questions'] = session['current_index']
            
            return session
            
        except Exception as e:
            logger.error(f"❌ 提交答案失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return session
    
    def next_question(self, session: Dict[str, Any]) -> Dict[str, Any]:
        """加载下一题"""
        session['current_index'] += 1
        
        if session['current_index'] <= len(session['questions']):
            session['current_question'] = session['questions'][session['current_index'] - 1]
            logger.info(f"📄 加载第 {session['current_index']} 题")
        
        return session
    
    def generate_report(self, session: Dict[str, Any]) -> str:
        """
        生成评估报告（使用盘古7B）
        """
        try:
            logger.info("📝 正在使用盘古7B生成智能评估报告...")
            
            report = self.evaluator.generate_comprehensive_report(
                session['student_id'],
                "综合评估",
                session['answer_records']
            )
            
            logger.info("✅ 盘古7B报告生成完成")
            return report
            
        except Exception as e:
            logger.error(f"❌ 生成报告失败: {e}")
            return f"报告生成失败: {str(e)}"
    
    # 辅助功能
    def import_questions(self, file_path: str) -> int:
        """导入题目"""
        count = self.question_db.import_from_json(file_path)
        # 重新构建索引
        if count > 0:
            logger.info("🔄 重新构建RAG索引...")
            all_questions = self.question_db.get_all_questions()
            self.rag_engine.build_question_index(all_questions)
        return count
    
    def get_database_statistics(self) -> Dict[str, Any]:
        """获取数据库统计"""
        return self.question_db.get_statistics()
    
    def get_system_info(self) -> str:
        """获取系统信息"""
        student_count = 0
        total_records = 0
        if self.bkt_algorithm and hasattr(self.bkt_algorithm, 'student_states'):
            student_count = len(self.bkt_algorithm.student_states)
            total_records = sum(
                sum(len(minor) for minor in major.values())
                for major in self.bkt_algorithm.student_states.values()
            )
        
        rag_stats = self.rag_engine.get_statistics() if self.rag_engine else {}
        
        info = f"""
系统版本: {self.config.SYSTEM_INFO['version']}
描述: {self.config.SYSTEM_INFO['description']}
模型: {self.config.SYSTEM_INFO['model']}
设备: {self.config.SYSTEM_INFO['device']}

核心技术:
  - 嵌入模型: BGE-small-zh-v1.5 (本地)
  - 语言模型: 盘古7B (本地，用于评估和选题)
  - 知识图谱: 本地RAG引擎 (向量检索 + 实体关系提取)
  - 学习建模: BKT算法 (贝叶斯知识追踪)
  - 自适应学习: 基于掌握度的动态难度调整

数据统计:
  - 题库路径: {self.config.QUESTION_DB}
  - 总题目数: {len(self.question_db.get_all_questions())}
  - RAG索引题目数: {rag_stats.get('total_questions', 0)}
  - 嵌入维度: {rag_stats.get('embedding_dim', 0)}
  - 学生数量: {student_count}
  - 学习记录数: {total_records}

智能功能:
  - 细粒度知识点追踪: ✅ 已启用
  - RAG向量检索: ✅ 已启用 (本地BGE)
  - 知识图谱构建: ✅ 已启用 (盘古7B实体提取)
  - 薄弱点自动识别: ✅ 已启用
  - 智能选题系统: ✅ 已启用 (RAG + BKT + 盘古7B + 多级备用)
  - 自适应难度调整: ✅ 已启用
  - AI答案评估: ✅ 已启用 (盘古7B)
  - AI报告生成: ✅ 已启用 (盘古7B)

NPU配置:
  - 可用NPU数量: {len(self.pangu_model.devices) if self.pangu_model else 0}
"""
        return info
    
    def reload_models(self):
        """重新加载模型"""
        if self.pangu_model:
            logger.info("🔄 重新加载盘古7B模型...")
            self.pangu_model.load_model()
    
    def clear_cache(self):
        """清除NPU缓存"""
        import torch
        try:
            import torch_npu
            if torch.npu.is_available():
                for i in range(torch.npu.device_count()):
                    torch.npu.empty_cache()
                logger.info("✅ NPU缓存已清除")
        except:
            pass


def create_system_core(config):
    """创建系统核心"""
    core = SmartEducationSystem(config)
    core.initialize()
    return core


if __name__ == "__main__":
    import sys
    sys.path.append("..")
    import config
    
    logging.basicConfig(level=logging.INFO)
    
    system = create_system_core(config)
    print("✅ 智能系统创建成功")
    print(system.get_system_info())