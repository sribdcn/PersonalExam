# -*- coding: utf-8 -*-
"""
Copyright (c) 2025 AI系统与应用课题组@SRIBD

基于LLM和知识图谱协同的个性化出题系统 (PersonalExam)
Personalized Question Generation System Based on LLM and Knowledge Graph Collaboration

学生评估模块 - 深度个性化版本
结合BKT算法生成真正反映学生特征的个性化评估报告
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import re
import hashlib
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


class PersonalizedStudentEvaluator:
    """个性化学生评估器"""
    
    def __init__(self, llm_model, bkt_algorithm, config: Dict[str, Any]):
        """
        初始化评估器
        
        Args:
            llm_model: 盘古7B模型实例
            bkt_algorithm: BKT算法实例
            config: 配置字典
        """
        self.llm_model = llm_model
        self.bkt_algorithm = bkt_algorithm
        self.config = config
        
        self.weight_difficulty = config.get('weight_difficulty', {
            '简单': 1.0,
            '中等': 1.5,
            '困难': 2.0
        })
        
        self.pass_score = config.get('pass_score', 0.6)
        self.excellent_score = config.get('excellent_score', 0.85)
        
        # 答案评估缓存（提升性能）
        self.answer_cache = {}
        self.cache_enabled = config.get('enable_answer_cache', True)
        self.cache_max_size = config.get('answer_cache_max_size', 1000)
        self.use_llm_evaluation = config.get('use_llm_evaluation', True)
        
        logger.info("✅ 个性化学生评估器初始化完成（带答案缓存）")
    
    def check_answer(self, question: Dict[str, Any],
                    student_answer: str,
                    prompt_template: str) -> Tuple[bool, str]:
        """检查学生答案是否正确（优化版：缓存+快速匹配）"""
        try:
            # 1) 先进行快速严格匹配：完全等价则直接返回，跳过LLM，显著降低延迟
            strict_ok = self._strict_answer_check(question, student_answer)
            if strict_ok:
                # 返回可解释理由（不调用LLM）
                return True, self._build_reason_for_strict(question, student_answer, True)

            cache_key: Optional[str] = None

            # 2) 检查缓存（仅当启用缓存时）
            if self.cache_enabled:
                cache_key = self._get_cache_key(question, student_answer)
                if cache_key in self.answer_cache:
                    logger.debug(f"✅ 答案评估缓存命中")
                    return self.answer_cache[cache_key]

            # 如果关闭LLM评估，直接返回严格判定结果
            if not self.use_llm_evaluation:
                result = (False, self._build_reason_for_strict(question, student_answer, False))
                if self.cache_enabled and cache_key:
                    self._add_to_cache(cache_key, result)
                return result

            # 3) 需要LLM参与的再调用模型（优化参数以提升速度）
            if not self.llm_model.is_loaded:
                logger.info("🔄 首次使用，正在加载盘古7B模型...")
                self.llm_model.load_model()
            
            logger.info("🤖 使用盘古7B模型进行智能答案评估（快速模式）")
            # 优化：进一步缩短生成长度、降低温度、关闭采样以提升速度
            prompt = prompt_template.format(
                question=question.get('问题', ''),
                correct_answer=question.get('答案', ''),
                student_answer=student_answer,
                explanation=question.get('解析', '')
            )
            
            response = self.llm_model.generate(
                prompt,
                temperature=0.05,  # 进一步降低温度，提升速度和稳定性
                top_p=0.9,
                max_length=128,  # 进一步缩短生成长度（只需要判定结果+简短理由）
                enable_thinking=False  # 关闭思维链，提升速度
            )
            
            is_correct, reason = self._parse_model_response(response)
            
            if is_correct is None:
                logger.warning("⚠️  模型响应不明确，使用备用严格判断逻辑")
                is_correct = self._strict_answer_check(question, student_answer)
                reason = self._build_reason_for_strict(question, student_answer, bool(is_correct))
            
            # 4) 缓存结果
            if self.cache_enabled and cache_key:
                self._add_to_cache(cache_key, (is_correct, reason))
            
            return is_correct, reason
            
        except Exception as e:
            logger.error(f"❌ 答案检查失败: {e}")
            is_correct = self._strict_answer_check(question, student_answer)
            # 使用规则化可解释理由（即使模型调用失败也要有详细理由）
            reason = self._build_reason_for_strict(question, student_answer, bool(is_correct))
            return is_correct, reason
    
    def _get_cache_key(self, question: Dict[str, Any], student_answer: str) -> str:
        """生成缓存键"""
        # 使用题目ID和答案的哈希值作为缓存键
        question_id = question.get('题号', '')
        answer_hash = hashlib.md5(
            (student_answer.lower().strip() + question.get('答案', '').lower().strip()).encode('utf-8')
        ).hexdigest()[:8]
        return f"{question_id}_{answer_hash}"
    
    def _add_to_cache(self, cache_key: str, result: Tuple[bool, str]):
        """添加到缓存"""
        # 限制缓存大小
        if len(self.answer_cache) >= self.cache_max_size:
            # 删除最旧的条目（简单FIFO策略）
            oldest_key = next(iter(self.answer_cache))
            del self.answer_cache[oldest_key]
        
        self.answer_cache[cache_key] = result
    
    def _parse_model_response(self, response: str) -> Tuple[bool, str]:
        """解析模型响应"""
        try:
            response = response.strip()
            
            result_pattern = r'判定结果[:：]\s*(正确|错误)'
            result_match = re.search(result_pattern, response, re.IGNORECASE)
            
            if result_match:
                result_text = result_match.group(1)
                is_correct = '正确' in result_text
                
                reason_pattern = r'理由[:：]\s*(.+?)(?:\n\n|\n判定|$)'
                reason_match = re.search(reason_pattern, response, re.DOTALL)
                reason = reason_match.group(1).strip() if reason_match else response
                
                return is_correct, reason
            
            response_lower = response.lower()
            correct_keywords = ['正确', '对的', '准确', '符合', '完整']
            incorrect_keywords = ['错误', '不对', '不正确', '不完整', '遗漏', '缺少']
            
            correct_count = sum(1 for kw in correct_keywords if kw in response_lower)
            incorrect_count = sum(1 for kw in incorrect_keywords if kw in response_lower)
            
            if incorrect_count > correct_count:
                return False, response
            elif correct_count > incorrect_count:
                if any(neg in response_lower for neg in ['不完整', '遗漏', '缺少', '部分']):
                    return False, response
                return True, response
            
            return None, response
            
        except Exception as e:
            logger.error(f"❌ 解析模型响应失败: {e}")
            return None, response
    
    def _strict_answer_check(self, question: Dict[str, Any], student_answer: str) -> bool:
        """备用严格答案检查逻辑（增强版，支持集合/顺序无关/间隔等价）"""
        correct_answer = (question.get('答案') or '').lower().strip()
        student_answer_lower = (student_answer or '').lower().strip()

        if not correct_answer or not student_answer_lower:
            return False

        # 1) 直接规范化字符串等价（去标点空白）
        correct_clean = re.sub(r'[\s\.,;!?，。；！？、]', '', correct_answer)
        student_clean = re.sub(r'[\s\.,;!?，。；！？、]', '', student_answer_lower)
        if correct_clean == student_clean:
            return True

        # 2) 解集/多解场景：按常见分隔符拆分并做集合比较（忽略顺序）
        def split_solutions(text: str):
            # 去掉变量名与等号，如 x=2 -> 2
            text = re.sub(r'[a-zA-Z]\s*=', '', text)
            parts = re.split(r'\s*(?:或|and|,|，|；|;|、|/|\bor\b|\||\s+或是\s+)\s*', text)
            parts = [p for p in parts if p]
            return parts

        corr_parts = split_solutions(correct_answer)
        stu_parts = split_solutions(student_answer_lower)

        # 如果标准答案明显包含多个部分（如“2 或 3”），进行集合等价判断
        if len(corr_parts) >= 2:
            # 优先用数值集合比较；若取不到数值，再用规范化文本集合比较
            corr_nums = self._extract_numbers(' '.join(corr_parts))
            stu_nums = self._extract_numbers(' '.join(stu_parts))
            if corr_nums:
                def to_float_list(nums):
                    vals = []
                    for n in nums:
                        try:
                            vals.append(float(n))
                        except Exception:
                            pass
                    return vals

                c_vals = to_float_list(corr_nums)
                s_vals = to_float_list(stu_nums)
                if c_vals and s_vals:
                    # 逐一匹配（容差）
                    unmatched = []
                    for c in c_vals:
                        if not any(abs(c - s) < 1e-2 for s in s_vals):
                            unmatched.append(c)
                    if not unmatched and len(s_vals) >= len(c_vals):
                        return True
            # 文本集合比较（去空白/标点）
            norm = lambda t: re.sub(r'[\s\.,;!?，。；！？、]', '', t)
            c_set = {norm(p) for p in corr_parts}
            s_set = {norm(p) for p in stu_parts}
            if c_set.issubset(s_set):
                return True

        # 3) 关键要点覆盖与区间处理
        key_info = self._extract_key_information(correct_answer)
        missing_info = []
        for info in key_info:
            if not self._contains_info(student_answer_lower, info):
                missing_info.append(info)
        if missing_info:
            return False

        # 4) 数值一致性（容差比较）
        correct_numbers = self._extract_numbers(correct_answer)
        student_numbers = self._extract_numbers(student_answer_lower)
        
        if correct_numbers:
            corr_vals = [float(num) for num in correct_numbers]
            stud_vals = [float(num) for num in student_numbers]
            matched_indices = set()
            for s_val in stud_vals:
                match_index = None
                for idx, c_val in enumerate(corr_vals):
                    if abs(c_val - s_val) < 0.01:
                        match_index = idx
                        break
                if match_index is None:
                    # 学生的数字在标准答案中不存在
                    return False
                matched_indices.add(match_index)
            # 情况1：全部数字都匹配
            numbers_ok = len(matched_indices) == len(corr_vals)
            # 情况2：只写出最终答案（匹配标准答案的最后一个数字）
            if not numbers_ok:
                if stud_vals and abs(stud_vals[-1] - corr_vals[-1]) < 0.01:
                    numbers_ok = True
            if not numbers_ok:
                return False
 
        if len(student_clean) < len(correct_clean) * 0.5:
            return False

        return True
    
    def _extract_key_information(self, text: str) -> List[str]:
        """提取关键信息"""
        key_info = []
        keywords = ['单调递增', '单调递减', '单调增加', '单调减少', '递增', '递减',
                   '最大值', '最小值', '极大值', '极小值', '或', '且']
        
        for keyword in keywords:
            if keyword in text:
                key_info.append(keyword)
        
        interval_patterns = [r'\([^)]+\)', r'\[[^\]]+\]', r'\([^)]+\)', r'\[[^\]]+\)']
        for pattern in interval_patterns:
            intervals = re.findall(pattern, text)
            key_info.extend(intervals)
        
        return key_info
    
    def _contains_info(self, text: str, info: str) -> bool:
        """检查文本是否包含信息"""
        text_clean = re.sub(r'[\s\.,;!?，。；！？、]', '', text)
        info_clean = re.sub(r'[\s\.,;!?，。；！？、]', '', info)
        return info_clean in text_clean
    
    def _extract_numbers(self, text: str) -> List[str]:
        """提取数字"""
        if not text:
            return []
        normalized = text
        for minus in ['−', '﹣', '–', '—', '―']:
            normalized = normalized.replace(minus, '-')
        normalized = normalized.replace(' ', '')
        return re.findall(r'-?\d+\.?\d*', normalized)

    # ==================== 规则化可解释理由（用于快速判定/回退） ====================
    def _normalize_text(self, text: str) -> str:
        normalized = (text or '').lower().strip()
        for minus in ['−', '﹣', '–', '—', '―']:
            normalized = normalized.replace(minus, '-')
        return re.sub(r'[\s\.,;!?，。；！？、]', '', normalized)

    def _numbers_diff(self, std_text: str, stu_text: str, tol: float = 1e-2):
        std_nums = self._extract_numbers(std_text)
        stu_nums = self._extract_numbers(stu_text)
        mismatches = []
        for s in std_nums:
            try:
                s_val = float(s)
            except Exception:
                continue
            if not any(abs(s_val - float(t)) < tol for t in stu_nums):
                mismatches.append(s)
        return mismatches, std_nums, stu_nums

    def _build_reason_for_strict(self, question: Dict[str, Any], student_answer: str, is_correct: bool) -> str:
        std = (question.get('答案') or '')
        std_norm = self._normalize_text(std)
        stu_norm = self._normalize_text(student_answer)

        # 关键点覆盖
        keys = self._extract_key_information(std)
        missing = [k for k in keys if not self._contains_info(student_answer.lower(), k)]
        hit = [k for k in keys if k not in missing]

        # 数值一致性
        num_miss, std_nums, stu_nums = self._numbers_diff(std, student_answer)

        # 相似度
        sim = 0.0
        try:
            sim = SequenceMatcher(None, std_norm, stu_norm).ratio()
        except Exception:
            pass

        lines = []
        if is_correct:
            if hit:
                lines.append(f"完整性: 覆盖关键要点（{', '.join(hit)}）")
            else:
                lines.append("完整性: 关键要点已覆盖")
            lines.append("准确性: 关键表达一致")
            if std_nums:
                lines.append(f"数值一致性: 一致（标准: {std_nums}，你的: {stu_nums}）")
            if sim:
                lines.append(f"文本相似度: {sim:.0%}")
        else:
            if missing:
                lines.append(f"缺失要点: {', '.join(missing)}")
            if num_miss:
                lines.append(f"数值不一致: 缺少/不匹配 {num_miss}（标准: {std_nums}，你的: {stu_nums}）")
            if sim and sim < 0.8:
                lines.append(f"文本相似度偏低: {sim:.0%}")
            if not lines:
                lines.append("与标准答案存在关键信息差异")
        return "；".join(lines)
    
    def analyze_learning_pattern(self, answer_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        分析学习模式（核心方法）
        
        Returns:
            学习模式分析结果
        """
        if not answer_records:
            return {}
        
        # 1. 答题速度分析（如果有时间戳）
        answer_pace = self._analyze_answer_pace(answer_records)
        
        # 2. 错误模式分析
        error_patterns = self._analyze_error_patterns(answer_records)
        
        # 3. 进步趋势分析
        progress_trend = self._analyze_progress_trend(answer_records)
        
        # 4. 稳定性分析
        stability = self._analyze_stability(answer_records)
        
        # 5. 难度适应性分析
        difficulty_adaptation = self._analyze_difficulty_adaptation(answer_records)
        
        return {
            'answer_pace': answer_pace,
            'error_patterns': error_patterns,
            'progress_trend': progress_trend,
            'stability': stability,
            'difficulty_adaptation': difficulty_adaptation
        }
    
    def _analyze_answer_pace(self, records: List[Dict[str, Any]]) -> str:
        """分析答题速度"""
        # 简化版本，可以扩展为基于时间戳的分析
        total_questions = len(records)
        if total_questions < 3:
            return "数据不足"
        
        # 基于掌握度变化速度推断
        mastery_changes = []
        for record in records:
            if 'mastery_change' in record:
                mastery_changes.append(abs(record['mastery_change']))
        
        if mastery_changes:
            avg_change = sum(mastery_changes) / len(mastery_changes)
            if avg_change > 0.15:
                return "快速反应型"
            elif avg_change > 0.08:
                return "稳健思考型"
            else:
                return "谨慎缓慢型"
        
        return "正常"
    
    def _analyze_error_patterns(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析错误模式"""
        wrong_records = [r for r in records if not r.get('is_correct', False)]
        
        if not wrong_records:
            return {'pattern': '无错误', 'details': []}
        
        # 按难度分组错误
        errors_by_difficulty = {'简单': 0, '中等': 0, '困难': 0}
        for record in wrong_records:
            diff = record.get('question', {}).get('难度', '中等')
            if diff in errors_by_difficulty:
                errors_by_difficulty[diff] += 1
        
        # 识别错误模式
        total_errors = len(wrong_records)
        simple_error_rate = errors_by_difficulty['简单'] / total_errors if total_errors > 0 else 0
        
        if simple_error_rate > 0.5:
            pattern = "基础薄弱型"
            description = "在简单题目上频繁出错，需要加强基础知识"
        elif errors_by_difficulty['困难'] > errors_by_difficulty['简单'] + errors_by_difficulty['中等']:
            pattern = "挑战困难型"
            description = "基础扎实，但在高难度题目上需要提升"
        else:
            pattern = "随机波动型"
            description = "错误分布较为均匀，需要提高整体稳定性"
        
        return {
            'pattern': pattern,
            'description': description,
            'errors_by_difficulty': errors_by_difficulty,
            'total_errors': total_errors
        }
    
    def _analyze_progress_trend(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析进步趋势"""
        if len(records) < 5:
            return {'trend': '数据不足', 'description': '需要更多答题数据'}
        
        # 分析掌握度变化趋势
        mastery_values = []
        for record in records:
            if 'mastery_after' in record:
                mastery_values.append(record['mastery_after'])
        
        if len(mastery_values) < 5:
            return {'trend': '数据不足', 'description': '需要更多答题数据'}
        
        # 计算前半部分和后半部分的平均掌握度
        mid = len(mastery_values) // 2
        first_half_avg = sum(mastery_values[:mid]) / mid
        second_half_avg = sum(mastery_values[mid:]) / (len(mastery_values) - mid)
        
        improvement = second_half_avg - first_half_avg
        
        if improvement > 0.15:
            trend = "快速进步"
            description = "学习能力强，掌握度显著提升"
        elif improvement > 0.05:
            trend = "稳步提升"
            description = "保持良好学习态势，持续进步"
        elif improvement > -0.05:
            trend = "基本稳定"
            description = "知识掌握相对稳定，可适当增加挑战"
        elif improvement > -0.15:
            trend = "轻微下降"
            description = "可能遇到学习瓶颈，需要调整学习策略"
        else:
            trend = "明显下降"
            description = "学习状态不佳，建议回顾基础知识"
        
        return {
            'trend': trend,
            'description': description,
            'improvement_value': improvement,
            'first_half_mastery': first_half_avg,
            'second_half_mastery': second_half_avg
        }
    
    def _analyze_stability(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析学习稳定性"""
        if len(records) < 5:
            return {'level': '数据不足', 'score': 0.5}
        
        # 分析答题结果的连续性
        results = [r.get('is_correct', False) for r in records]
        
        # 计算连续性：相邻答题结果相同的比例
        continuity = sum(1 for i in range(1, len(results)) if results[i] == results[i-1])
        continuity_rate = continuity / (len(results) - 1) if len(results) > 1 else 0
        
        # 计算掌握度波动
        mastery_values = [r.get('mastery_after', 0.5) for r in records if 'mastery_after' in r]
        if len(mastery_values) > 2:
            mastery_std = self._calculate_std(mastery_values)
        else:
            mastery_std = 0
        
        # 综合评分
        stability_score = 0.6 * (1 - mastery_std) + 0.4 * continuity_rate
        
        if stability_score > 0.75:
            level = "非常稳定"
            description = "学习状态稳定，表现可预测"
        elif stability_score > 0.55:
            level = "基本稳定"
            description = "学习状态较为稳定，偶有波动"
        elif stability_score > 0.35:
            level = "波动较大"
            description = "学习状态起伏明显，需要调整节奏"
        else:
            level = "极不稳定"
            description = "学习状态波动剧烈，建议寻找原因"
        
        return {
            'level': level,
            'description': description,
            'score': stability_score,
            'mastery_std': mastery_std,
            'continuity_rate': continuity_rate
        }
    
    def _calculate_std(self, values: List[float]) -> float:
        """计算标准差"""
        if not values:
            return 0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def _analyze_difficulty_adaptation(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析难度适应性"""
        difficulty_performance = {'简单': [], '中等': [], '困难': []}
        
        for record in records:
            diff = record.get('question', {}).get('难度', '中等')
            is_correct = record.get('is_correct', False)
            if diff in difficulty_performance:
                difficulty_performance[diff].append(is_correct)
        
        # 计算各难度准确率
        accuracy_by_diff = {}
        for diff, results in difficulty_performance.items():
            if results:
                accuracy_by_diff[diff] = sum(results) / len(results)
            else:
                accuracy_by_diff[diff] = 0
        
        # 判断适应性类型
        simple_acc = accuracy_by_diff.get('简单', 0)
        medium_acc = accuracy_by_diff.get('中等', 0)
        hard_acc = accuracy_by_diff.get('困难', 0)
        
        if simple_acc > 0.8 and medium_acc > 0.6 and hard_acc > 0.4:
            adaptation_type = "全面型"
            description = "各难度题目适应良好，学习能力均衡"
        elif simple_acc > 0.9 and hard_acc < 0.3:
            adaptation_type = "基础型"
            description = "擅长简单题目，需要逐步提升挑战难度"
        elif hard_acc > 0.5 and simple_acc < 0.7:
            adaptation_type = "跳跃型"
            description = "能应对难题但基础不够扎实，建议巩固基础"
        else:
            adaptation_type = "发展型"
            description = "正在适应不同难度，继续保持练习"
        
        return {
            'type': adaptation_type,
            'description': description,
            'accuracy_by_difficulty': accuracy_by_diff,
            'strength_level': max(accuracy_by_diff, key=accuracy_by_diff.get) if accuracy_by_diff else '中等'
        }
    
    def generate_personalized_portrait(self, student_id: str, 
                                      knowledge_point: str,
                                      answer_records: List[Dict[str, Any]]) -> str:
        """
        生成个性化学生画像（文本描述）
        
        Args:
            student_id: 学生ID
            knowledge_point: 知识点
            answer_records: 答题记录
            
        Returns:
            个性化画像文本
        """
        # 1. 获取BKT学生档案
        student_profile = self.bkt_algorithm.generate_student_profile(student_id)
        
        # 2. 分析学习模式
        learning_pattern = self.analyze_learning_pattern(answer_records)
        
        # 3. 生成画像描述
        portrait = f"""
╔══════════════════════════════════════════════════════════════╗
║                   个性化学生画像                              ║
║                  Student ID: {student_id:20s}          ║
╚══════════════════════════════════════════════════════════════╝

【基本信息】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  学生ID: {student_id}
  评估知识点: {knowledge_point}
  累计学习: {student_profile.get('total_knowledge_points', 0)} 个知识点
  累计答题: {student_profile.get('total_answers', 0)} 题
  整体掌握度: {student_profile.get('overall_mastery', 0):.1%}

【学习能力画像】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        # 学习潜力
        learning_potential = student_profile.get('learning_potential', '未知')
        portrait += f"  🎯 学习潜力: {learning_potential}\n"
        
        # 学习特征
        if 'learning_characteristics' in student_profile:
            char = student_profile['learning_characteristics']
            portrait += f"  📊 难度偏好: {char.get('difficulty_preference', '中等')}\n"
            portrait += f"  💎 学习稳定性: {char.get('learning_stability', 0):.1%}\n"
            
            if learning_pattern.get('answer_pace'):
                portrait += f"  ⚡ 答题风格: {learning_pattern['answer_pace']}\n"
        
        # 进步趋势
        if 'progress_trend' in learning_pattern:
            trend = learning_pattern['progress_trend']
            portrait += f"  📈 进步趋势: {trend.get('trend', '未知')}\n"
            portrait += f"     {trend.get('description', '')}\n"
        
        portrait += "\n【知识掌握情况】\n"
        portrait += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        
        # 优势和薄弱点
        strengths = student_profile.get('strengths', [])
        weak_points = student_profile.get('weak_points', [])
        
        if strengths:
            portrait += f"  ✅ 优势知识点:\n"
            for point in strengths:
                kp_data = student_profile['knowledge_points'].get(point, {})
                mastery = kp_data.get('mastery', 0)
                portrait += f"     • {point}: {mastery:.1%}\n"
        else:
            portrait += f"  ✅ 优势知识点: 暂无明显优势（继续加油）\n"
        
        if weak_points:
            portrait += f"\n  ⚠️  薄弱知识点:\n"
            for point in weak_points:
                kp_data = student_profile['knowledge_points'].get(point, {})
                mastery = kp_data.get('mastery', 0)
                portrait += f"     • {point}: {mastery:.1%} ← 需要重点加强\n"
        else:
            portrait += f"\n  ⚠️  薄弱知识点: 无明显薄弱环节（表现均衡）\n"
        
        # 本次测评表现
        portrait += "\n【本次测评分析】\n"
        portrait += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        
        # 难度适应性
        if 'difficulty_adaptation' in learning_pattern:
            adapt = learning_pattern['difficulty_adaptation']
            portrait += f"  🎪 适应性类型: {adapt.get('type', '未知')}\n"
            portrait += f"     {adapt.get('description', '')}\n"
            portrait += f"  💪 最强难度: {adapt.get('strength_level', '未知')}\n"
            
            if 'accuracy_by_difficulty' in adapt:
                portrait += f"\n  各难度表现:\n"
                for diff, acc in adapt['accuracy_by_difficulty'].items():
                    bar = self._create_progress_bar(acc)
                    portrait += f"     {diff:4s} {bar} {acc:.1%}\n"
        
        # 错误模式
        if 'error_patterns' in learning_pattern:
            error = learning_pattern['error_patterns']
            portrait += f"\n  🔍 错误模式: {error.get('pattern', '未知')}\n"
            portrait += f"     {error.get('description', '')}\n"
        
        # 稳定性分析
        if 'stability' in learning_pattern:
            stability = learning_pattern['stability']
            portrait += f"\n  🎯 学习稳定性: {stability.get('level', '未知')}\n"
            portrait += f"     {stability.get('description', '')}\n"
        
        portrait += "\n"
        
        return portrait
    
    def _create_progress_bar(self, value: float, length: int = 20) -> str:
        """创建进度条"""
        filled = int(value * length)
        bar = '█' * filled + '░' * (length - filled)
        return f"[{bar}]"
    
    def generate_ai_recommendations(self, student_id: str,
                                   knowledge_point: str,
                                   answer_records: List[Dict[str, Any]],
                                   learning_pattern: Dict[str, Any]) -> str:
        """
        使用AI生成个性化学习建议
        
        Args:
            student_id: 学生ID
            knowledge_point: 知识点
            answer_records: 答题记录
            learning_pattern: 学习模式分析结果
            
        Returns:
            AI生成的个性化建议
        """
        try:
            if not self.llm_model.is_loaded:
                self.llm_model.load_model()
            
            # 构建详细的上下文信息
            context = self._build_recommendation_context(
                student_id, knowledge_point, answer_records, learning_pattern
            )
            
            prompt = f"""你是一位经验丰富的教育专家和学习顾问。请基于以下学生的详细学习数据，生成一份深度个性化的学习建议报告。

{context}

请生成包含以下内容的个性化学习建议：

1. **学习优势分析**（2-3条）
   - 识别学生的学习优势
   - 说明这些优势如何帮助学习

2. **改进重点**（3-4条）
   - 明确指出需要改进的方面
   - 每条建议要具体、可操作

3. **学习方法建议**（3-4条）
   - 根据学生的学习风格推荐学习方法
   - 提供具体的练习建议

4. **短期目标**（1-2周内）
   - 设定2-3个可实现的小目标
   - 说明如何检验目标完成情况

5. **长期规划**（1-2个月）
   - 提出整体学习方向
   - 建议下一步学习的知识点

请用友好、鼓励的语气，让学生感受到支持和信心。建议要具体、可操作，避免空泛的表述。"""

            logger.info("🤖 正在使用盘古7B生成深度个性化学习建议...")
            recommendations = self.llm_model.generate(prompt, temperature=0.7)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ 生成AI建议失败: {e}")
            return self._generate_fallback_recommendations(learning_pattern)
    
    def _build_recommendation_context(self, student_id: str,
                                     knowledge_point: str,
                                     answer_records: List[Dict[str, Any]],
                                     learning_pattern: Dict[str, Any]) -> str:
        """构建推荐上下文"""
        # 获取学生档案
        student_profile = self.bkt_algorithm.generate_student_profile(student_id)
        
        # 计算统计数据
        total_questions = len(answer_records)
        correct_count = sum(1 for r in answer_records if r.get('is_correct', False))
        accuracy = correct_count / total_questions if total_questions > 0 else 0
        
        # 掌握度变化
        if answer_records and 'mastery_before' in answer_records[0]:
            initial_mastery = answer_records[0].get('mastery_before', 0.3)
            final_mastery = answer_records[-1].get('mastery_after', 0.3)
            mastery_change = final_mastery - initial_mastery
        else:
            initial_mastery = 0.3
            final_mastery = 0.3
            mastery_change = 0
        
        context = f"""
【学生基本信息】
- 学生ID: {student_id}
- 评估知识点: {knowledge_point}
- 学习潜力: {student_profile.get('learning_potential', '未知')}
- 整体掌握度: {student_profile.get('overall_mastery', 0):.1%}

【本次测评数据】
- 答题总数: {total_questions}
- 正确题数: {correct_count}
- 准确率: {accuracy:.1%}
- 初始掌握度: {initial_mastery:.1%}
- 最终掌握度: {final_mastery:.1%}
- 掌握度提升: {mastery_change:+.1%}

【学习风格特征】"""

        # 添加学习模式信息
        if 'answer_pace' in learning_pattern:
            context += f"\n- 答题风格: {learning_pattern['answer_pace']}"
        
        if 'stability' in learning_pattern:
            stability = learning_pattern['stability']
            context += f"\n- 学习稳定性: {stability.get('level', '未知')} ({stability.get('description', '')})"
        
        if 'difficulty_adaptation' in learning_pattern:
            adapt = learning_pattern['difficulty_adaptation']
            context += f"\n- 适应性类型: {adapt.get('type', '未知')} ({adapt.get('description', '')})"
            context += f"\n- 最擅长难度: {adapt.get('strength_level', '未知')}"
        
        if 'progress_trend' in learning_pattern:
            trend = learning_pattern['progress_trend']
            context += f"\n- 进步趋势: {trend.get('trend', '未知')} ({trend.get('description', '')})"
        
        if 'error_patterns' in learning_pattern:
            error = learning_pattern['error_patterns']
            context += f"\n- 错误模式: {error.get('pattern', '未知')} ({error.get('description', '')})"
        
        # 添加优势和薄弱点
        strengths = student_profile.get('strengths', [])
        weak_points = student_profile.get('weak_points', [])
        
        if strengths:
            context += f"\n\n【优势知识点】\n- " + "\n- ".join(strengths)
        
        if weak_points:
            context += f"\n\n【薄弱知识点】\n- " + "\n- ".join(weak_points)
        
        return context
    
    def _generate_fallback_recommendations(self, learning_pattern: Dict[str, Any]) -> str:
        """生成备用建议（当AI不可用时）"""
        recommendations = "\n【个性化学习建议】\n"
        recommendations += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        
        # 根据学习模式给出建议
        if 'error_patterns' in learning_pattern:
            error = learning_pattern['error_patterns']
            recommendations += f"1. 针对你的 '{error.get('pattern', '')}' 特点:\n"
            recommendations += f"   {error.get('description', '')}\n\n"
        
        if 'progress_trend' in learning_pattern:
            trend = learning_pattern['progress_trend']
            recommendations += f"2. 学习趋势建议:\n"
            recommendations += f"   {trend.get('description', '')}\n\n"
        
        if 'difficulty_adaptation' in learning_pattern:
            adapt = learning_pattern['difficulty_adaptation']
            recommendations += f"3. 难度调整建议:\n"
            recommendations += f"   {adapt.get('description', '')}\n\n"
        
        recommendations += "4. 通用建议:\n"
        recommendations += "   - 每天坚持练习，保持学习的连续性\n"
        recommendations += "   - 及时复习错题，总结解题方法\n"
        recommendations += "   - 循序渐进，不要急于求成\n"
        
        return recommendations
    
    def generate_comprehensive_report(self, student_id: str,
                                     knowledge_point: str,
                                     answer_records: List[Dict[str, Any]]) -> str:
        """
        生成综合个性化评估报告（完整版）
        
        Args:
            student_id: 学生ID
            knowledge_point: 知识点
            answer_records: 答题记录
            
        Returns:
            完整的个性化评估报告
        """
        logger.info(f"📝 正在生成学生 {student_id} 的综合个性化评估报告...")
        
        # 1. 生成个性化画像
        portrait = self.generate_personalized_portrait(student_id, knowledge_point, answer_records)
        
        # 2. 分析学习模式
        learning_pattern = self.analyze_learning_pattern(answer_records)
        
        # 3. 生成AI个性化建议
        ai_recommendations = self.generate_ai_recommendations(
            student_id, knowledge_point, answer_records, learning_pattern
        )
        
        # 4. 组合完整报告
        report = portrait
        report += ai_recommendations
        
        # 5. 添加掌握度变化图表
        report += self._generate_mastery_trend_chart(answer_records)
        
        # 6. 添加结束语
        report += "\n\n" + "="*64 + "\n"
        report += "💡 温馨提示: 学习是一个持续的过程，保持耐心和恒心最重要！\n"
        report += "📞 如有疑问，欢迎随时向老师咨询。加油！💪\n"
        report += "="*64 + "\n"
        
        logger.info("✅ 综合个性化评估报告生成完成")
        
        return report
    
    def _generate_mastery_trend_chart(self, answer_records: List[Dict[str, Any]]) -> str:
        """生成掌握度变化趋势图（文本版）"""
        if not answer_records or 'mastery_after' not in answer_records[0]:
            return ""
        
        chart = "\n【掌握度变化趋势】\n"
        chart += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        
        # 提取掌握度数据
        mastery_values = [r.get('mastery_after', 0) for r in answer_records]
        
        # 生成ASCII图表
        max_width = 50
        chart += "\n"
        for i, mastery in enumerate(mastery_values, 1):
            bar_length = int(mastery * max_width)
            bar = '█' * bar_length
            result_symbol = '✓' if answer_records[i-1].get('is_correct') else '✗'
            chart += f"  Q{i:2d} {result_symbol} {bar} {mastery:.1%}\n"
        
        # 添加趋势说明
        if len(mastery_values) > 1:
            trend = mastery_values[-1] - mastery_values[0]
            chart += f"\n  总体趋势: "
            if trend > 0.1:
                chart += f"显著上升 ↗ (+{trend:.1%})"
            elif trend > 0:
                chart += f"稳步上升 ↗ (+{trend:.1%})"
            elif trend > -0.1:
                chart += f"基本稳定 → ({trend:+.1%})"
            else:
                chart += f"有所下降 ↘ ({trend:+.1%})"
        
        chart += "\n"
        
        return chart
    
    def calculate_score(self, answer_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算得分"""
        total_weight = 0
        earned_weight = 0
        correct_count = 0
        
        difficulty_stats = {
            '简单': {'total': 0, 'correct': 0},
            '中等': {'total': 0, 'correct': 0},
            '困难': {'total': 0, 'correct': 0}
        }
        
        for record in answer_records:
            question = record['question']
            is_correct = record['is_correct']
            difficulty = question.get('难度', '中等')
            
            weight = self.weight_difficulty.get(difficulty, 1.0)
            total_weight += weight
            
            if is_correct:
                earned_weight += weight
                correct_count += 1
            
            if difficulty in difficulty_stats:
                difficulty_stats[difficulty]['total'] += 1
                if is_correct:
                    difficulty_stats[difficulty]['correct'] += 1
        
        total_score = (earned_weight / total_weight * 100) if total_weight > 0 else 0
        accuracy = (correct_count / len(answer_records) * 100) if answer_records else 0
        
        for diff, stats in difficulty_stats.items():
            if stats['total'] > 0:
                stats['accuracy'] = stats['correct'] / stats['total'] * 100
            else:
                stats['accuracy'] = 0
        
        return {
            'total_score': round(total_score, 2),
            'accuracy': round(accuracy, 2),
            'correct_count': correct_count,
            'total_count': len(answer_records),
            'difficulty_stats': difficulty_stats
        }


def create_evaluator(llm_model, bkt_algorithm, config: Dict[str, Any]):
    """
    创建个性化评估器
    
    Args:
        llm_model: 盘古模型实例
        bkt_algorithm: BKT算法实例
        config: 配置字典
        
    Returns:
        个性化评估器实例
    """
    return PersonalizedStudentEvaluator(llm_model, bkt_algorithm, config)


if __name__ == "__main__":
    # 测试代码
    import sys
    sys.path.append("..")
    from config import (PANGU_MODEL_PATH, EVALUATION_MODEL_CONFIG,
                       EVALUATION_CONFIG, PROMPTS)
    from models import create_llm_model
    from utils.bkt_algorithm import create_bkt_algorithm
    
    logging.basicConfig(level=logging.INFO)
    
    # 创建模型和算法
    pangu_model = create_llm_model('pangu', PANGU_MODEL_PATH, EVALUATION_MODEL_CONFIG)
    bkt = create_bkt_algorithm()
    
    # 创建评估器
    evaluator = create_evaluator(pangu_model, bkt, EVALUATION_CONFIG)
    
    print("✅ 个性化评估器测试完成")