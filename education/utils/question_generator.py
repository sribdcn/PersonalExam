"""
出题生成模块 - 增强版
使用盘古7B模型结合LightRAG和知识图谱生成题目
"""

import json
import logging
import random
import re
from typing import List, Dict, Any, Optional
import asyncio

logger = logging.getLogger(__name__)


class EnhancedQuestionGenerator:
    """增强版题目生成器 - 使用盘古7B和LightRAG"""
    
    def __init__(self, llm_model, question_db, rag_engine, config: Dict[str, Any]):
        self.llm_model = llm_model
        self.question_db = question_db
        self.rag_engine = rag_engine
        self.config = config
        
        logger.info("✅ 增强版题目生成器初始化完成（盘古7B + LightRAG）")
    
    async def get_reference_from_rag(self, knowledge_point: str, 
                                    difficulty: str = None,
                                    count: int = 3) -> str:
        """从RAG系统检索参考题目"""
        try:
            # 构建查询
            query = f"关于{knowledge_point}的题目"
            if difficulty:
                query += f"，难度为{difficulty}"
            
            # 查询RAG
            logger.info(f"🔍 从知识图谱检索: {query}")
            rag_result = await self.rag_engine.query(query, mode="hybrid")
            
            # 同时从题库检索
            db_questions = self.question_db.get_questions_by_minor_point(
                knowledge_point.split('/')[0] if '/' in knowledge_point else knowledge_point,
                knowledge_point.split('/')[1] if '/' in knowledge_point else ''
            )
            
            if difficulty:
                # 筛选难度
                diff_map = {'简单': (0.0, 0.35), '中等': (0.35, 0.65), '困难': (0.65, 1.0)}
                if difficulty in diff_map:
                    low, high = diff_map[difficulty]
                    db_questions = [q for q in db_questions 
                                  if low <= q.get('难度', 0.5) < high]
            
            # 随机选择参考题目
            if len(db_questions) > count:
                db_questions = random.sample(db_questions, count)
            
            # 格式化参考示例
            reference_text = self._format_reference_examples(db_questions)
            
            # 如果RAG返回有用信息，添加到参考中
            if rag_result and "模拟" not in rag_result:
                reference_text += f"\n\n### RAG检索结果:\n{rag_result[:500]}"
            
            return reference_text
            
        except Exception as e:
            logger.error(f"❌ RAG检索失败: {e}")
            # 降级为仅从题库检索
            return self._get_reference_from_db(knowledge_point, difficulty, count)
    
    def _get_reference_from_db(self, knowledge_point: str, 
                              difficulty: str = None,
                              count: int = 3) -> str:
        """从题库检索参考题目（降级方案）"""
        # 先按知识点筛选
        if '/' in knowledge_point:
            major, minor = knowledge_point.split('/')
            questions = self.question_db.get_questions_by_minor_point(major.strip(), minor.strip())
        else:
            questions = self.question_db.get_questions_by_major_point(knowledge_point)
        
        # 如果指定难度，进一步筛选
        if difficulty:
            diff_map = {'简单': (0.0, 0.35), '中等': (0.35, 0.65), '困难': (0.65, 1.0)}
            if difficulty in diff_map:
                low, high = diff_map[difficulty]
                questions = [q for q in questions if low <= q.get('难度', 0.5) < high]
        
        # 随机选择
        if len(questions) > count:
            questions = random.sample(questions, count)
        
        return self._format_reference_examples(questions)
    
    def _format_reference_examples(self, questions: List[Dict[str, Any]]) -> str:
        """格式化参考示例"""
        if not questions:
            return "无参考示例"
        
        examples = []
        for i, q in enumerate(questions, 1):
            example = f"""
示例{i}:
问题: {q.get('问题', '')}
答案: {q.get('答案', '')}
解析: {q.get('解析', '')}
难度: {q.get('难度', 0.5)}
"""
            examples.append(example.strip())
        
        return "\n\n".join(examples)
    
    def _build_generation_prompt(self, knowledge_point: str,
                                 difficulty: str,
                                 reference_text: str) -> str:
        """构建生成题目的提示词"""
        
        prompt = f"""你是一位经验丰富的数学教师，擅长出题。请根据以下要求生成一道高质量的数学题目。

【生成要求】
1. 知识点: {knowledge_point}
2. 难度等级: {difficulty}
3. 题目类型: 计算题或应用题

【参考示例】
{reference_text}

【输出格式】
请严格按照以下JSON格式输出，不要添加任何其他内容：

{{
  "问题": "题目描述",
  "答案": "标准答案",
  "解析": "详细解题步骤",
  "难度": "难度值(0-1之间的小数)",
  "知识点大类": "知识点大类名称",
  "知识点小类": "知识点小类名称"
}}

【重要提示】
- 题目要有明确的问题和答案
- 解析要详细清晰，便于学生理解
- 难度要符合要求（简单: 0.0-0.35, 中等: 0.35-0.65, 困难: 0.65-1.0）
- 题目要原创，不要直接复制参考示例
- 输出必须是合法的JSON格式

请直接输出JSON，不要有任何前后文字说明。
"""
        return prompt
    
    async def generate_single_question(self, knowledge_point: str,
                                      difficulty: str = "中等",
                                      max_retries: int = 3) -> Optional[Dict[str, Any]]:
        """
        生成单个题目
        
        Args:
            knowledge_point: 知识点（可以是"大类/小类"格式）
            difficulty: 难度等级
            max_retries: 最大重试次数
        """
        for attempt in range(max_retries):
            try:
                logger.info(f"🤖 正在使用盘古7B生成题目 (尝试 {attempt+1}/{max_retries})...")
                
                # 1. 从RAG检索参考
                reference_text = await self.get_reference_from_rag(
                    knowledge_point, difficulty, count=2
                )
                
                # 2. 构建提示词
                prompt = self._build_generation_prompt(
                    knowledge_point, difficulty, reference_text
                )
                
                # 3. 确保盘古模型已加载
                if not self.llm_model.is_loaded:
                    logger.info("📥 首次使用，正在加载盘古7B模型...")
                    self.llm_model.load_model()
                
                # 4. 调用盘古7B生成
                logger.info("🔄 盘古7B正在生成题目...")
                response = self.llm_model.generate(
                    prompt, 
                    temperature=0.8,  # 提高创造性
                    max_length=2048
                )
                
                logger.info(f"📝 盘古7B响应: {response[:200]}...")
                
                # 5. 解析响应
                question = self._parse_generated_question(response)
                
                if question:
                    # 确保知识点字段正确
                    if '/' in knowledge_point:
                        major, minor = knowledge_point.split('/')
                        question['knowledge_point_major'] = major.strip()
                        question['knowledge_point_minor'] = minor.strip()
                        question['知识点大类'] = major.strip()
                        question['知识点小类'] = minor.strip()
                    
                    logger.info("✅ 题目生成成功（盘古7B）")
                    return question
                else:
                    logger.warning(f"⚠️  题目解析失败，重试中...")
                    
            except Exception as e:
                logger.error(f"❌ 生成题目出错: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        logger.error(f"❌ 生成题目失败，已尝试 {max_retries} 次")
        return None
    
    def _parse_generated_question(self, response: str) -> Optional[Dict[str, Any]]:
        """解析盘古7B生成的题目"""
        try:
            # 1. 尝试找到JSON部分
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                logger.error("❌ 响应中未找到JSON格式")
                return self._extract_question_from_text(response)
            
            json_str = response[start_idx:end_idx]
            
            # 2. 尝试直接解析
            try:
                question = json.loads(json_str)
                logger.info("✅ JSON解析成功")
            except json.JSONDecodeError as e:
                logger.warning(f"⚠️  直接JSON解析失败: {e}，尝试修复")
                
                # 修复常见JSON格式问题
                json_str = re.sub(r'(\w+):', r'"\1":', json_str)  # 修复缺少引号的键
                json_str = json_str.replace("'", '"')  # 单引号转双引号
                
                try:
                    question = json.loads(json_str)
                    logger.info("✅ 修复后JSON解析成功")
                except json.JSONDecodeError as e2:
                    logger.warning(f"⚠️  修复后仍失败: {e2}，提取关键信息")
                    return self._extract_question_from_text(response)
            
            # 3. 验证必要字段
            required_fields = ['问题', '答案', '解析']
            for field in required_fields:
                if field not in question or not question[field]:
                    logger.error(f"❌ 缺少必要字段: {field}")
                    return None
            
            # 4. 确保有难度值
            if '难度' not in question:
                question['难度'] = 0.5
            elif isinstance(question['难度'], str):
                # 如果是字符串，尝试转换
                try:
                    question['难度'] = float(question['难度'])
                except:
                    question['难度'] = 0.5
            
            return question
            
        except Exception as e:
            logger.error(f"❌ 解析题目失败: {e}")
            return None
    
    def _extract_question_from_text(self, response: str) -> Optional[Dict[str, Any]]:
        """从文本中提取题目信息（后备方案）"""
        try:
            question = {}
            
            # 提取各个字段
            patterns = {
                '问题': r'问题[:：]\s*([^\n]+)',
                '答案': r'答案[:：]\s*([^\n]+)',
                '解析': r'解析[:：]\s*([^\n]+)',
                '难度': r'难度[:：]\s*([^\n]+)',
                '知识点': r'知识点[:：]\s*([^\n]+)'
            }
            
            for field, pattern in patterns.items():
                match = re.search(pattern, response, re.MULTILINE)
                if match:
                    question[field] = match.group(1).strip()
            
            # 检查是否提取到足够信息
            if len(question) >= 3:
                logger.info("✅ 从文本提取题目信息成功")
                
                # 处理难度
                if '难度' in question:
                    try:
                        question['难度'] = float(question['难度'])
                    except:
                        question['难度'] = 0.5
                
                return question
            else:
                logger.error("❌ 提取的题目信息不完整")
                return None
                
        except Exception as e:
            logger.error(f"❌ 文本提取失败: {e}")
            return None
    
    def generate_question_set(self, knowledge_point: str,
                            count: int,
                            difficulty_distribution: Dict[str, float] = None) -> List[Dict[str, Any]]:
        """
        生成题目集合（同步包装）
        
        Args:
            knowledge_point: 知识点
            count: 题目数量
            difficulty_distribution: 难度分布
        """
        # 运行异步函数
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                self._async_generate_question_set(
                    knowledge_point, count, difficulty_distribution
                )
            )
            return result
        finally:
            loop.close()
    
    async def _async_generate_question_set(self, knowledge_point: str,
                                          count: int,
                                          difficulty_distribution: Dict[str, float] = None) -> List[Dict[str, Any]]:
        """异步生成题目集合"""
        if difficulty_distribution is None:
            difficulty_distribution = {'简单': 0.3, '中等': 0.5, '困难': 0.2}
        
        # 计算每个难度的题目数量
        difficulty_counts = {}
        remaining = count
        
        for difficulty, ratio in difficulty_distribution.items():
            num = int(count * ratio)
            difficulty_counts[difficulty] = num
            remaining -= num
        
        # 剩余的分配给中等难度
        if remaining > 0:
            difficulty_counts['中等'] = difficulty_counts.get('中等', 0) + remaining
        
        # 生成题目
        generated_questions = []
        
        for difficulty, num in difficulty_counts.items():
            logger.info(f"📝 正在生成 {num} 道{difficulty}难度的题目...")
            
            for i in range(num):
                question = await self.generate_single_question(
                    knowledge_point=knowledge_point,
                    difficulty=difficulty
                )
                
                if question:
                    generated_questions.append(question)
                    logger.info(f"✅ 进度: {len(generated_questions)}/{count}")
                else:
                    logger.warning(f"⚠️  生成第{i+1}题失败，跳过")
        
        logger.info(f"🎉 题目生成完成，成功 {len(generated_questions)}/{count} 道")
        return generated_questions


class MockQuestionGenerator:
    """模拟题目生成器（从题库抽取）"""
    
    def __init__(self, llm_model, question_db, rag_engine, config: Dict[str, Any]):
        self.question_db = question_db
        self.config = config
        logger.info("⚠️  使用模拟题目生成器（从题库抽取）")
    
    async def generate_single_question(self, knowledge_point: str, 
                                      difficulty: str = "中等") -> Optional[Dict[str, Any]]:
        """从题库抽取题目"""
        if '/' in knowledge_point:
            major, minor = knowledge_point.split('/')
            questions = self.question_db.get_questions_by_minor_point(
                major.strip(), minor.strip()
            )
        else:
            questions = self.question_db.get_questions_by_major_point(knowledge_point)
        
        # 筛选难度
        diff_map = {'简单': (0.0, 0.35), '中等': (0.35, 0.65), '困难': (0.65, 1.0)}
        if difficulty in diff_map:
            low, high = diff_map[difficulty]
            questions = [q for q in questions if low <= q.get('难度', 0.5) < high]
        
        if questions:
            return random.choice(questions)
        return None
    
    def generate_question_set(self, knowledge_point: str, count: int, **kwargs):
        """生成题目集"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            questions = []
            for _ in range(count):
                q = loop.run_until_complete(
                    self.generate_single_question(knowledge_point)
                )
                if q:
                    questions.append(q)
            return questions
        finally:
            loop.close()


def create_question_generator(llm_model, question_db, rag_engine, config: Dict[str, Any],
                             use_real_generation: bool = True):
    """
    创建题目生成器
    
    Args:
        llm_model: 盘古7B模型
        question_db: 题库
        rag_engine: RAG引擎
        config: 配置
        use_real_generation: 是否使用真实生成（False则从题库抽取）
    """
    if use_real_generation:
        return EnhancedQuestionGenerator(llm_model, question_db, rag_engine, config)
    return MockQuestionGenerator(llm_model, question_db, rag_engine, config)


if __name__ == "__main__":
    # 测试代码
    import sys
    sys.path.append("..")
    from config import (PANGU_MODEL_PATH, QUESTION_MODEL_CONFIG,
                       QUESTION_DB, LIGHTRAG_CONFIG)
    from models import create_llm_model
    from data_management.question_db import create_question_database
    from knowledge_management.rag_engine import create_rag_engine
    from models.embedding_model import create_embedding_model, lightrag_embedding_func
    
    logging.basicConfig(level=logging.INFO)
    
    # 创建组件
    pangu_model = create_llm_model('pangu', PANGU_MODEL_PATH, QUESTION_MODEL_CONFIG)
    question_db = create_question_database(str(QUESTION_DB))
    
    # 创建嵌入模型和RAG
    embedding_model = create_embedding_model(
        "/home/weitianyu/bgem3",
        {"device": "cpu", "batch_size": 32}
    )
    rag_engine = create_rag_engine(
        LIGHTRAG_CONFIG,
        lambda texts: lightrag_embedding_func(texts, embedding_model)
    )
    
    # 创建生成器
    generator = create_question_generator(
        pangu_model, question_db, rag_engine, {}, use_real_generation=True
    )
    
    # 测试生成
    async def test():
        question = await generator.generate_single_question("代数/一元二次方程", "简单")
        if question:
            print(f"\n生成的题目:\n{json.dumps(question, ensure_ascii=False, indent=2)}")
    
    asyncio.run(test())