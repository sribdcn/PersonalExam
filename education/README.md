# 🎓 智能教育评估对话系统

基于LightRAG知识图谱和AI大模型的智能教育评估系统,支持自动出题、智能评估和知识图谱可视化。

## 📋 项目特性

- ✅ **智能出题**: 使用Qwen模型结合知识图谱自动生成题目
- ✅ **智能评估**: 使用PanGu模型评估学生答题表现
- ✅ **知识图谱**: 基于LightRAG构建题目知识图谱
- ✅ **图谱可视化**: 使用NetworkX和Plotly进行交互式可视化
- ✅ **题库管理**: 支持老师导入、添加、管理题目
- ✅ **友好界面**: 基于Gradio的Web界面,易于使用

## 🏗️ 系统架构

```
edu_assessment_system/
├── config.py                 # 系统配置
├── system_core.py           # 系统核心
├── main.py                  # 主程序入口
├── requirements.txt         # 依赖包
├── models/                  # 模型模块
│   ├── llm_models.py       # LLM模型封装
│   └── embedding_model.py  # 嵌入模型封装
├── lightrag_module/        # LightRAG模块
│   └── rag_engine.py       # RAG引擎
├── data_management/        # 数据管理模块
│   └── question_db.py      # 题库管理
├── utils/                  # 工具模块
│   ├── question_generator.py  # 出题生成器
│   └── evaluator.py           # 学生评估器
├── visualization/          # 可视化模块
│   └── kg_visualizer.py    # 知识图谱可视化
├── ui/                     # UI界面模块
│   └── main_ui.py          # Gradio界面
├── data/                   # 数据目录
│   └── question_database.json  # 题库文件
├── rag_storage/            # RAG存储目录
└── logs/                   # 日志目录
```

## 🚀 快速开始

### 1. 环境准备

**系统要求**:
- Python 3.10+
- CUDA (可选,用于GPU加速)
- 16GB+ RAM

### 2. 安装依赖

```bash
# 克隆或解压项目
cd edu_assessment_system

# 安装依赖包
pip install -r requirements.txt

# 如果需要从源码安装LightRAG
pip install git+https://github.com/HKUDS/LightRAG.git
```

### 3. 模型准备

下载并放置以下模型到指定位置:

1. **PanGu模型** (用于评估)
   - 下载地址: [openPangu-Embedded-7B-V1.1](https://huggingface.co/models)
   - 放置位置: `D:\HUAWEI\pangu` (Windows)

2. **Qwen模型** (用于出题)
   - 下载地址: [Qwen3.0系列](https://huggingface.co/Qwen)
   - 推荐: Qwen-7B-Chat 或 Qwen-14B-Chat
   - 放置位置: `D:\HUAWEI\qwen` (Windows)

3. **BGE-M3嵌入模型** (用于知识图谱)
   - 下载地址: [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)
   - 放置位置: `D:\HUAWEI\bgem3` (Windows)

> 💡 **提示**: 如果模型文件不存在,系统会自动以模拟模式运行,用于测试和演示。

### 4. 配置系统

编辑 `config.py` 文件,修改模型路径(如果需要):

```python
# Windows路径示例
PANGU_MODEL_PATH = r"D:\HUAWEI\pangu"
QWEN_MODEL_PATH = r"D:\HUAWEI\qwen"
BGE_M3_MODEL_PATH = r"D:\HUAWEI\bgem3"

# Linux路径示例
# PANGU_MODEL_PATH = "/home/user/models/pangu"
# QWEN_MODEL_PATH = "/home/user/models/qwen"
# BGE_M3_MODEL_PATH = "/home/user/models/bge-m3"
```

### 5. 导入示例数据

将提供的 `math.json` 文件复制到 `data/` 目录:

```bash
cp /path/to/math.json data/
```

### 6. 启动系统

```bash
python main.py
```

系统启动后,浏览器会自动打开 `http://localhost:7860`

## 📖 使用指南

### 学生测评流程

1. **选择知识点**: 从下拉菜单选择要测试的知识点(如"代数"、"几何"等)
2. **设置参数**: 
   - 选择难度偏好(简单/中等/困难/混合)
   - 设置题目数量(3-15题)
3. **开始答题**: 点击"开始测评"按钮
4. **作答题目**: 在答案框输入答案,点击"提交答案"
5. **查看反馈**: 系统会显示正确答案和解析
6. **继续答题**: 点击"下一题"继续
7. **查看报告**: 完成所有题目后,查看详细评估报告

### 教师管理功能

#### 导入题目

1. 进入"教师管理"标签页
2. 选择"导入题目"子标签
3. 上传JSON格式的题库文件
4. 点击"导入"按钮

**JSON格式示例**:
```json
[
  {
    "问题": "求解方程 x^2 - 5x + 6 = 0",
    "答案": "x = 2 或 x = 3",
    "解析": "因式分解: (x-2)(x-3) = 0",
    "难度": "简单",
    "知识点": "代数"
  }
]
```

#### 添加单题

1. 选择"添加单题"子标签
2. 填写题目信息(知识点、难度、题目、答案、解析)
3. 点击"添加题目"按钮

#### 查看题库

1. 选择"查看题库"子标签
2. 点击"刷新统计"查看题库概况
3. 使用筛选器搜索特定题目

### 知识图谱可视化

1. 进入"知识图谱"标签页
2. 选择布局算法:
   - `spring`: 弹簧布局(默认)
   - `circular`: 环形布局
   - `kamada_kawai`: 力导向布局
3. 点击"生成图谱"按钮
4. 可以下载HTML文件保存图谱

**图谱节点说明**:
- 🔴 红色菱形: 知识点节点
- 🔵 蓝色方形: 难度节点
- 🟢 绿色圆形: 题目节点

## 🔧 配置说明

### 核心配置 (`config.py`)

```python
# 出题配置
QUESTION_CONFIG = {
    "questions_per_type": 5,  # 每个知识点默认出题数量
    "min_difficulty": "简单",
    "max_difficulty": "困难"
}

# 评估配置
EVALUATION_CONFIG = {
    "pass_score": 0.6,      # 及格分数(60%)
    "excellent_score": 0.85,  # 优秀分数(85%)
    "weight_difficulty": {
        "简单": 1.0,
        "中等": 1.5,
        "困难": 2.0
    }
}

# UI配置
UI_CONFIG = {
    "title": "智能教育评估对话系统",
    "port": 7860,
    "share": False  # 设为True可以生成公网链接
}
```

## 📊 系统功能详解

### 1. 智能出题系统

- **基于知识图谱**: 利用LightRAG分析题库,生成相关题目
- **难度控制**: 支持简单、中等、困难三个难度级别
- **参考示例**: 自动从题库中提取相似题目作为参考
- **LLM生成**: 使用Qwen模型生成高质量题目

### 2. 智能评估系统

- **答案检查**: 使用PanGu模型智能判断答案正确性
- **综合评分**: 考虑难度权重的加权计分
- **能力分析**: 分析不同难度的答题表现
- **薄弱点识别**: 自动识别学生的薄弱知识点
- **个性化建议**: 生成针对性的学习建议

### 3. 知识图谱系统

- **自动构建**: 从题目数据自动构建知识图谱
- **关系挖掘**: 发现知识点之间的关联
- **交互可视化**: 支持缩放、拖拽、查看详情
- **多种布局**: 支持多种图谱布局算法

## 🐛 故障排除

### 常见问题

1. **模型加载失败**
   - 检查模型路径是否正确
   - 确认模型文件完整
   - 检查内存是否充足

2. **CUDA错误**
   - 在 `config.py` 中将 `device` 改为 `"cpu"`
   - 或安装对应版本的CUDA

3. **题目生成失败**
   - 检查Qwen模型是否正常加载
   - 确认题库中有参考题目
   - 查看日志文件获取详细错误

4. **可视化无法显示**
   - 确认安装了plotly库
   - 检查浏览器是否支持WebGL
   - 尝试更换布局算法

### 日志查看

日志文件位置: `logs/system.log`

```bash
# 查看最新日志
tail -f logs/system.log

# 搜索错误信息
grep ERROR logs/system.log
```

## 🔄 更新日志

### v1.0.0 (2025-10-23)
- ✅ 初始版本发布
- ✅ 支持智能出题和评估
- ✅ 知识图谱可视化
- ✅ 题库管理功能
- ✅ Gradio Web界面

## 📝 TODO

- [ ] 支持更多题型(选择题、填空题等)
- [ ] 添加学生历史记录功能
- [ ] 支持多学科扩展
- [ ] 优化知识图谱算法
- [ ] 添加数据导出功能
- [ ] 支持多用户管理

## 📄 许可证

本项目仅供学习和研究使用。

## 🤝 贡献

欢迎提交Issue和Pull Request!

## 📧 联系方式

如有问题,请通过以下方式联系:
- 提交GitHub Issue
- 发送邮件至: [your-email]

---

**祝使用愉快! 🎉**
