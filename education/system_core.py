"""
系统核心模块 - 智能个性化版本（增强版）
集成RAG引擎，使用盘古7B进行智能出题和评估
"""

import logging
import random
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
import asyncio

logger = logging.getLogger(__name__)


class SmartEducationSystem:
    """智能教育评估系统核心（增强版）"""
    
    def __init__(self, config):
        self.config = config
        self.question_db = None
        self.embedding_model = None
        self.pangu_model = None
        self.evaluator = None
        self.visualizer = None
        self.bkt_algorithm = None
        self.rag_engine = None
        self.question_generator = None
        self.models_loaded = False
        
        logger.info("✅ 智能教育系统核心初始化（增强版）")
    
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
            from models.embedding_model import lightrag_embedding_func
            from utils.question_generator import create_question_generator  # 修复：改为 utils
            
            # 1. 初始化题库
            logger.info("📚 初始化题库...")
            self.question_db = create_question_database(str(self.config.QUESTION_DB))
            
            # 2. 初始化嵌入模型
            logger.info("🔤 初始化嵌入模型...")
            self.embedding_model = create_embedding_model(
                self.config.BGE_M3_MODEL_PATH,
                self.config.EMBEDDING_MODEL_CONFIG
            )
            
            # 3. 初始化RAG引擎
            logger.info("🧠 初始化RAG引擎...")
            self.rag_engine = create_rag_engine(
                self.config.LIGHTRAG_CONFIG,
                lambda texts: lightrag_embedding_func(texts, self.embedding_model)
            )
            
            # 异步初始化RAG
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.rag_engine.initialize())
                logger.info("✅ RAG引擎初始化成功")
                
                # 构建知识图谱
                logger.info("🔄 正在构建知识图谱...")
                from knowledge_management.rag_engine import QuestionRAGManager
                rag_manager = QuestionRAGManager(self.rag_engine)
                questions = self.question_db.get_all_questions()
                loop.run_until_complete(rag_manager.build_kg_from_questions(questions))
                logger.info("✅ 知识图谱构建完成")
            finally:
                loop.close()
            
            # 4. 初始化盘古模型
            logger.info("🚀 初始化盘古7B模型...")
            self.pangu_model = create_llm_model(
                'pangu',
                self.config.PANGU_MODEL_PATH,
                self.config.PANGU_MODEL_CONFIG
            )
            
            logger.info("🔄 预加载盘古7B模型...")
            self.pangu_model.load_model()
            logger.info("✅ 盘古7B模型加载完成")
            
            # 5. 初始化BKT算法
            logger.info("🧠 初始化BKT算法...")
            self.bkt_algorithm = create_bkt_algorithm(
                storage_path=str(self.config.DATA_DIR / "student_states.json")
            )
            
            # 6. 初始化评估器（使用盘古7B）
            logger.info("📊 初始化评估器（盘古7B驱动）...")
            self.evaluator = create_evaluator(
                self.pangu_model,
                self.bkt_algorithm,
                self.config.EVALUATION_CONFIG
            )
            
            # 7. 初始化题目生成器（使用盘古7B + RAG）
            logger.info("📝 初始化题目生成器（盘古7B + LightRAG）...")
            self.question_generator = create_question_generator(
                self.pangu_model,
                self.question_db,
                self.rag_engine,
                self.config.SMART_QUESTION_CONFIG,
                use_real_generation=True  # 使用真实生成
            )
            
            # 8. 初始化可视化
            logger.info("🎨 初始化可视化组件...")
            self.visualizer = create_visualizer(
                self.config.VISUALIZATION_CONFIG
            )
            
            self.models_loaded = True
            logger.info("✅ 系统初始化完成 - 智能个性化自适应学习版（盘古7B驱动）")
            
        except Exception as e:
            logger.error(f"❌ 系统初始化失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise RuntimeError(f"系统初始化失败: {e}")
    
    def _analyze_student_weakness(self, student_id: str) -> List[Tuple[str, str]]:
        """
        分析学生薄弱知识点
        
        Returns:
            List of (major_point, minor_point) tuples
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
        
        return [(major, minor) for major, minor, _ in weak_points]
    
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
    
    def _select_target_knowledge_point(self, student_id: str, 
                                       weak_point_ratio: float = 0.7) -> Tuple[str, str]:
        """
        智能选择目标知识点
        
        Args:
            student_id: 学生ID
            weak_point_ratio: 选择薄弱点的概率
            
        Returns:
            (major_point, minor_point)
        """
        # 获取薄弱知识点
        weak_points = self._analyze_student_weakness(student_id)
        
        # 获取未探索知识点
        unexplored_points = self._get_unexplored_points(student_id)
        
        # 决策：薄弱点 vs 探索新知识点
        if weak_points and random.random() < weak_point_ratio:
            # 优先加强薄弱点
            selected = weak_points[0]  # 选择最薄弱的
            logger.info(f"🎯 选择薄弱知识点: {selected[0]}/{selected[1]}")
            return selected
        elif unexplored_points:
            # 探索新知识点
            selected = random.choice(unexplored_points)
            logger.info(f"🔍 探索新知识点: {selected[0]}/{selected[1]}")
            return selected
        else:
            # 随机选择一个知识点
            all_kp = self.question_db.get_all_knowledge_points()
            major = random.choice(list(all_kp.keys()))
            minor = random.choice(all_kp[major])
            logger.info(f"🎲 随机选择知识点: {major}/{minor}")
            return major, minor
    
    def _select_question_by_mastery(self, student_id: str, major_point: str,
                                   minor_point: str, used_ids: set) -> Optional[Dict[str, Any]]:
        """
        根据掌握度选择题目（基于BKT算法）
        
        Args:
            student_id: 学生ID
            major_point: 知识点大类
            minor_point: 知识点小类
            used_ids: 已使用的题目ID
            
        Returns:
            选中的题目
        """
        # 获取该知识点的所有题目
        candidates = self.question_db.get_questions_by_minor_point(major_point, minor_point)
        
        # 过滤已使用的题目
        candidates = [q for q in candidates if q.get('题号') not in used_ids]
        
        if not candidates:
            logger.warning(f"⚠️  知识点 {major_point}/{minor_point} 无可用题目")
            return None
        
        # 获取学生当前掌握度（BKT算法）
        state = self.bkt_algorithm.get_student_state(student_id, major_point, minor_point)
        mastery = state.mastery_prob
        
        # 根据掌握度确定难度范围（自适应）
        if mastery < 0.3:
            # 基础薄弱 - 选择简单题
            difficulty_range = (0.0, 0.4)
            logger.debug(f"🎯 掌握度 {mastery:.3f} - 自适应选择简单题")
        elif mastery < 0.7:
            # 中等水平 - 选择中等题
            difficulty_range = (0.3, 0.7)
            logger.debug(f"🎯 掌握度 {mastery:.3f} - 自适应选择中等题")
        else:
            # 掌握良好 - 选择困难题
            difficulty_range = (0.6, 1.0)
            logger.debug(f"🎯 掌握度 {mastery:.3f} - 自适应选择困难题")
        
        # 筛选合适难度的题目
        suitable = [q for q in candidates 
                   if difficulty_range[0] <= q.get('难度', 0.5) < difficulty_range[1]]
        
        if suitable:
            selected = random.choice(suitable)
        else:
            # 如果没有合适难度的题目，随机选一个
            logger.warning(f"⚠️  无合适难度题目，随机选择")
            selected = random.choice(candidates)
        
        logger.info(f"✅ 选中题目 {selected.get('题号')} (难度: {selected.get('难度', 0.5):.2f})")
        return selected
    
    def start_smart_assessment(self, student_id: str = "default_student",
                              num_questions: int = 10) -> Optional[Dict[str, Any]]:
        """
        开始智能测评（基于BKT算法的自适应测评）
        
        Args:
            student_id: 学生ID
            num_questions: 题目数量
            
        Returns:
            会话状态
        """
        try:
            logger.info(f"🚀 开始智能测评: 学生 {student_id}, 题数 {num_questions}")
            logger.info(f"📊 使用BKT算法进行自适应题目选择...")
            
            # 分析学生情况
            profile = self.bkt_algorithm.generate_student_profile(student_id)
            
            # 安全地访问字段
            total_kp = profile.get('total_knowledge_points', 0)
            overall_mastery = profile.get('overall_mastery', 0.0)
            
            logger.info(f"📊 学生档案: 整体掌握度 {overall_mastery:.3f}, "
                       f"已学知识点 {total_kp}")
            
            # 选择第一个目标知识点（智能推荐）
            major_point, minor_point = self._select_target_knowledge_point(student_id)
            
            # 选择第一题（基于掌握度）
            used_ids = set()
            first_question = self._select_question_by_mastery(
                student_id, major_point, minor_point, used_ids
            )
            
            if not first_question:
                logger.error("❌ 无法选择第一题")
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
            
            logger.info(f"✅ 测评开始 - 第1题: {major_point}/{minor_point}")
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
        
        关键：这里使用盘古7B进行答案评估
        """
        try:
            question = session['current_question']
            major_point = session['current_major_point']
            minor_point = session['current_minor_point']
            
            logger.info(f"✍️  评估答案 (题目 {session['current_index']}/{session['total_questions']})")
            logger.info(f"🤖 使用盘古7B进行严格答案评估...")
            
            # 使用盘古7B检查答案（核心功能）
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
            
            # 如果还有后续题目，智能选择下一题
            if session['current_index'] < session['total_questions']:
                logger.info(f"🤔 基于BKT算法智能选择下一题...")
                
                # 选择下一个目标知识点
                next_major, next_minor = self._select_target_knowledge_point(
                    session['student_id']
                )
                
                # 选择题目（基于更新后的掌握度）
                next_question = self._select_question_by_mastery(
                    session['student_id'],
                    next_major,
                    next_minor,
                    session['used_question_ids']
                )
                
                if next_question:
                    session['questions'].append(next_question)
                    session['used_question_ids'].add(next_question.get('题号'))
                    session['current_major_point'] = next_major
                    session['current_minor_point'] = next_minor
                    logger.info(f"✅ 准备下一题: {next_major}/{next_minor}")
                else:
                    logger.warning("⚠️  无法选择下一题，提前结束")
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
        
        关键：这里使用盘古7B生成个性化报告
        """
        try:
            logger.info("📝 正在使用盘古7B生成智能评估报告...")
            logger.info("🤖 盘古7B将分析学习模式并生成个性化建议...")
            
            # 使用盘古7B生成个性化报告（核心功能）
            report = self.evaluator.generate_comprehensive_report(
                session['student_id'],
                "综合评估",  # 不再限定单一知识点
                session['answer_records']
            )
            
            logger.info("✅ 盘古7B报告生成完成")
            return report
            
        except Exception as e:
            logger.error(f"❌ 生成报告失败: {e}")
            return f"报告生成失败: {str(e)}"
    
    # 以下是辅助功能
    def import_questions(self, file_path: str) -> int:
        """导入题目"""
        return self.question_db.import_from_json(file_path)
    
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
        
        info = f"""
系统版本: {self.config.SYSTEM_INFO['version']}
描述: {self.config.SYSTEM_INFO['description']}
模型: {self.config.SYSTEM_INFO['model']}
设备: {self.config.SYSTEM_INFO['device']}

核心技术:
  - 语言模型: 盘古7B (用于答案评估和报告生成)
  - 知识图谱: LightRAG (用于题目检索和生成)
  - 学习建模: BKT算法 (贝叶斯知识追踪)
  - 自适应学习: 基于掌握度的动态难度调整

数据统计:
  - 题库路径: {self.config.QUESTION_DB}
  - 总题目数: {len(self.question_db.get_all_questions())}
  - 学生数量: {student_count}
  - 学习记录数: {total_records}

智能功能:
  - 细粒度知识点追踪: ✅ 已启用
  - 薄弱点自动识别: ✅ 已启用
  - 智能选题系统: ✅ 已启用 (BKT + RAG)
  - 自适应难度调整: ✅ 已启用
  - AI答案评估: ✅ 已启用 (盘古7B)
  - AI报告生成: ✅ 已启用 (盘古7B)
  - 知识图谱检索: ✅ 已启用 (LightRAG)
"""
        return info
    
    def reload_models(self):
        """重新加载模型"""
        if self.pangu_model:
            logger.info("🔄 重新加载盘古7B模型...")
            self.pangu_model.load_model()
    
    def clear_cache(self):
        """清除缓存"""
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