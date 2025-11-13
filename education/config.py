# -*- coding: utf-8 -*-
"""
Copyright (c) 2025 AI系统与应用课题组@SRIBD

基于LLM和知识图谱协同的个性化出题系统 (PersonalExam)
Personalized Question Generation System Based on LLM and Knowledge Graph Collaboration

基于LLM和知识图谱协同的个性化出题系统配置文件 - 本地RAG版本
移除LightRAG依赖，使用本地向量检索
"""

import os
from pathlib import Path

# ==================== 项目路径配置 ====================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
QUESTION_DB = DATA_DIR / "question_database_2.json"
KG_GRAPH_PATH = DATA_DIR / "knowledge_graph.html"

# 确保目录存在
DATA_DIR.mkdir(exist_ok=True)

# ==================== 模型路径配置 ====================
PANGU_MODEL_PATH = os.getenv(
    "PANGU_MODEL_PATH",
    str(MODELS_DIR / "openPangu-Embedded-7B-V1.1")  
)

BGE_M3_MODEL_PATH = os.getenv(
    "BGE_MODEL_PATH",
    str(MODELS_DIR / "bge-small-zh-v1.5")   
)  

# ==================== 盘古7B模型配置 ====================
PANGU_MODEL_CONFIG = {
    "model_path": PANGU_MODEL_PATH,
    "max_new_tokens": 32768,
    "temperature": 0.7,
    "top_p": 0.9,
    "device": "npu",
    "torch_dtype": "float16",
    "trust_remote_code": True,
    "eos_token_id": 45892,
    "system_prompt": "你必须严格遵守法律法规和社会道德规范。生成任何内容时，都应避免涉及暴力、色情、恐怖主义、种族歧视、性别歧视等不当内容。一旦检测到输入或输出有此类倾向,应拒绝回答并发出警告。",
}

# 出题模型配置
QUESTION_MODEL_CONFIG = PANGU_MODEL_CONFIG.copy()
QUESTION_MODEL_CONFIG.update({
    "temperature": 0.8,
    "top_p": 0.95,
})

# 评估模型配置
EVALUATION_MODEL_CONFIG = PANGU_MODEL_CONFIG.copy()
EVALUATION_MODEL_CONFIG.update({
    "temperature": 0.3,
    "top_p": 0.85,
})

# 嵌入模型配置（仅使用BGE）
EMBEDDING_MODEL_CONFIG = {
    "model_path": BGE_M3_MODEL_PATH,
    "device": "cpu",
    "batch_size": 32,
    "max_length": 512,
    "use_pangu_embedding": False,  # 不使用盘古嵌入
}

# ==================== 知识点配置（新结构） ====================
# 知识点层级结构：大类 -> 小类
KNOWLEDGE_HIERARCHY = {
    "代数": [
        "根式方程", "递推数列", "指数方程", "对数方程", 
        "等比数列", "等差数列", "一元二次方程", "高次方程",
        "分式不等式", "对数不等式", "指数不等式", "二次不等式",
        "集合运算", "函数定义域", "最值问题", "绝对值函数",
        "复数运算", "数列求和"
    ],
    "几何": [
        "空间向量", "圆锥曲线", "空间直线与平面", 
        "立体几何", "平面直线", "直线与曲线"
    ],
    "分析": [
        "函数极值", "函数单调性", "极限计算", "定积分",
        "驻点求解", "导数几何意义", "基本导数"
    ],
    "概率统计": [
        "几何概率", "组合计数", "古典概率"
    ]
}

# 难度等级映射（数值 -> 文字描述）
DIFFICULTY_MAPPING = {
    (0.0, 0.35): "简单",
    (0.35, 0.65): "中等", 
    (0.65, 1.0): "困难"
}

def get_difficulty_label(difficulty_value: float) -> str:
    """根据数值获取难度标签"""
    for (low, high), label in DIFFICULTY_MAPPING.items():
        if low <= difficulty_value < high:
            return label
    return "中等"  # 默认

# ==================== 智能出题配置 ====================
SMART_QUESTION_CONFIG = {
    "default_question_count": 10,  # 默认题目数量
    "min_question_count": 5,       # 最少题目数
    "max_question_count": 20,      # 最多题目数
    
    # 薄弱点识别阈值
    "weak_threshold": 0.4,         # 掌握度低于此值视为薄弱
    "strong_threshold": 0.7,       # 掌握度高于此值视为掌握良好
    
    # 题目选择策略
    "weak_point_focus_ratio": 0.7, # 70%题目来自薄弱知识点
    "exploration_ratio": 0.3,      # 30%题目用于探索新知识点
    
    # 难度调整参数
    "difficulty_adjust_step": 0.1, # 难度调整步长
    
    # 知识点轮询
    "weak_point_rotation": True,   # 是否轮询薄弱知识点
    
    # RAG检索参数
    "rag_top_k": 5,               # RAG检索返回的题目数量
    
    # 题目选择优化
    "use_llm_selector": False,     # 是否使用LLM进行题目选择（默认关闭以提升速度）
    "use_rag_selector": False,     # 是否使用RAG向量检索（默认关闭以提升速度）
    "question_radar_default": 0.5, # 雷达图默认难度/掌握度（无数据时）
}

# ==================== 评估配置 ====================
EVALUATION_CONFIG = {
    "pass_score": 0.6,
    "excellent_score": 0.85,
    "enable_thinking": False,
    "enable_answer_cache": True,  # 启用答案评估缓存
    "answer_cache_max_size": 1000,  # 缓存最大条目数
    "use_llm_evaluation": False,   # 默认关闭LLM评估，加速判题
}

# ==================== 可视化配置 ====================
VISUALIZATION_CONFIG = {
    "node_size": 3000,
    "node_color": "lightblue",
    "edge_color": "gray",
    "font_size": 10,
    "figure_size": (15, 10),
    "layout": "spring"
}

# ==================== UI配置 ====================
UI_CONFIG = {
    "title": "基于LLM和知识图谱协同的个性化出题系统",
    "theme": "default",
    "port": 7860,
    "share": False,
    "server_name": "0.0.0.0"
}

# ==================== Prompt模板 ====================
PROMPTS = {
    "answer_check": """你是一名严谨、专业的数学老师，需要对学生答案进行严格评判。

题目: {question}
标准答案: {correct_answer}
学生答案: {student_answer}
解析: {explanation}

【重要评判标准】:
1. **完整性检查**：学生答案必须包含标准答案的所有关键信息点
2. **准确性检查**：学生答案的每个信息点都必须正确
3. **等价性判断**：只有当学生答案在数学上完全等价于标准答案时才判正确

请按照以下格式严格回答：
判定结果: [正确/错误]
理由: [详细说明判定理由]

不要添加任何其他内容。
""",

    "evaluation": """你是一个专业的教育评估专家。请根据学生的答题情况给出全面、专业的综合评价。

学生信息:
- 学生ID: {student_id}
- 答题数量: {total_questions}
- 正确数量: {correct_count}
- 准确率: {accuracy:.1f}%

答题详情:
{answer_details}

请从以下几个方面进行专业评估:
1. 知识掌握程度评级
2. 答题表现分析
3. 薄弱知识点识别
4. 个性化学习建议

请用专业、友好的语气给出详细评价。
"""
}

# ==================== 日志配置 ====================
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "handlers": ["console", "file"],
    "log_file": str(PROJECT_ROOT / "logs" / "system.log")
}

(PROJECT_ROOT / "logs").mkdir(exist_ok=True)

# ==================== 系统信息 ====================
SYSTEM_INFO = {
    "version": "3.1.0",
    "author": "AI系统及应用课题组@SRIBD",
    "description": "基于LLM和知识图谱协同的个性化出题系统（基于向量检索和知识图谱）",
    "model": "openPanGu-Embedded-7B-V1.1",
    "device": "Ascend 910B NPU",
    "rag_engine": "Local Vector Search + Knowledge Graph"
}

if __name__ == "__main__":
    print("配置文件加载成功!")
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"数据目录: {DATA_DIR}")
    print(f"题库文件: {QUESTION_DB}")
    print(f"知识点层级: {len(KNOWLEDGE_HIERARCHY)} 个大类")
    total_subpoints = sum(len(v) for v in KNOWLEDGE_HIERARCHY.values())
    print(f"知识点小类总数: {total_subpoints}")
    print(f"RAG引擎: 本地向量检索 + 知识图谱")