# 🧠 基于LLM和知识图谱协同的个性化出题系统

**Personalized Question Generation System Based on LLM and Knowledge Graph Collaboration**

[![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)](https://github.com/sribdcn/PersonalExam)
[![Python](https://img.shields.io/badge/python-3.11.12-green.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-BSL%201.1-orange.svg)](LICENSE)

一个基于LLM和知识图谱协同的个性化出题系统，使用BKT算法、RAG引擎和OpenPangu模型，实现个性化的智能题目生成和推荐。

**开发单位**: 深圳市大数据研究院 (SRIBD) | **课题组**: AI系统与应用课题组

**Powered by OpenPangu. OpenPangu is a trademark of Huawei Technologies Co., Ltd.**

**许可证**: 
- **项目代码**: 本项目代码采用 **BSL 1.1 (Business Source License 1.1)** 许可证，允许非商业使用，商业使用需要授权。
- **OpenPangu模型**: 本项目使用的OpenPangu模型采用 OPENPANGU MODEL LICENSE AGREEMENT VERSION 1.0 许可证，需遵守该许可协议的所有条款。

## 📋 目录

- [功能特性](#-功能特性)
- [技术架构](#-技术架构)
- [系统要求](#-系统要求)
  - [硬件要求](#硬件要求)
  - [软件要求](#软件要求)
  - [部署环境](#部署环境)
- [配置说明](#️-配置说明)
- [使用指南](#-使用指南)
- [项目结构](#-项目结构)
- [开源许可证](#-开源许可证)
- [贡献指南](#-贡献指南)
- [常见问题](#-常见问题)
- [更新日志](#-更新日志)
- [相关资源](#-相关资源)
- [免责声明](#免责声明)
- [反馈](#反馈)

## ✨ 功能特性

### 🎯 智能测评
- **自适应题目推荐**: 基于BKT算法和RAG引擎，根据学生掌握度智能推荐题目
- **细粒度知识点追踪**: 支持知识点大类和小类两级精细追踪
- **实时掌握度更新**: 使用贝叶斯知识追踪算法实时更新学生掌握度
- **AI答案评估**: 使用盘古7B模型进行智能答案评估，提供详细反馈

### 📊 学习分析
- **学生画像生成**: 自动生成个性化的学生学习画像
- **薄弱点识别**: 自动识别学生薄弱知识点，提供针对性建议
- **学习模式分析**: 分析答题速度、错误模式、进步趋势等
- **可视化报告**: 生成详细的学习分析报告和可视化图表

### 🕸️ 知识图谱
- **关系网络可视化**: 展示题目、知识点和难度之间的关系网络
- **多种布局算法**: 支持spring、circular、kamada_kawai等布局
- **交互式探索**: 基于Plotly的交互式知识图谱可视化

### ⚙️ 系统管理
- **题库管理**: 支持JSON格式题目导入和管理
- **系统监控**: 实时监控系统状态和资源使用
- **模型管理**: 支持模型重载和缓存清理

## 🏗️ 技术架构

### 核心技术栈


### 系统架构图

```mermaid
%%{init: {'theme':'base', 'themeVariables': {'primaryBackgroundColor':'#ffffff', 'primaryTextColor':'#000000', 'primaryBorderColor':'#333333', 'lineColor':'#333333', 'edgeLabelBackground':'#ffffff', 'secondaryColor':'#f0f0f0', 'tertiaryColor':'#ffffff', 'mainBkgColor':'#ffffff', 'secondBkgColor':'#f5f5f5', 'tertiaryBkgColor':'#ffffff', 'background':'#ffffff', 'primaryColor':'#ffffff'}}}%%
graph TB
    subgraph 前端层["前端层"]
        GradioUI["Gradio Web UI<br/>enhanced_main_ui.py<br/>交互式Web界面"]
        Login["登录注册系统<br/>用户认证与权限管理"]
    end
    
    subgraph 应用层["应用层"]
        SystemCore["System Core<br/>system_core_db.py<br/>系统核心协调器"]
        BKT["BKT算法适配器<br/>bkt_database_adapter.py<br/>贝叶斯知识追踪"]
        RAG["RAG引擎<br/>rag_engine.py<br/>基于知识图谱的检索"]
        KG["知识图谱构建器<br/>kg_builder.py<br/>实体关系提取与图谱构建"]
        Selector["题目选择器<br/>question_generator.py<br/>智能推荐+题目互选"]
        Evaluator["评估器<br/>evaluator.py<br/>答案评判+学习分析+报告生成"]
        Visualizer["可视化器<br/>kg_visualizer.py<br/>radar_chart.py<br/>知识图谱+雷达图可视化"]
    end
    
    subgraph AI模型层["AI模型层"]
        Pangu["OpenPangu 7B<br/>llm_models.py<br/>• 实体关系提取<br/>• 题目互选<br/>• 答案评判<br/>• 分析总结"]
        BGE["BGE-small-zh-v1.5<br/>embedding_model.py<br/>文本向量化"]
    end
    
    subgraph 数据层["数据层"]
        DB[("SQLite数据库<br/>database.py<br/>• 用户表<br/>• 题目表<br/>• 答题记录表<br/>• 学生状态表")]
        Cache[("知识图谱缓存<br/>knowledge_graph.pkl<br/>实体关系图")]
        Hash[("知识图谱哈希<br/>kg_hash.txt<br/>检测题库更新")]
        VectorIndex[("向量索引<br/>内存存储<br/>题目向量检索")]
    end
    
    subgraph 硬件层["硬件层"]
        NPU["昇腾910B2 NPU<br/>AI模型加速"]
    end
    
    %% 前端到核心
    GradioUI --> Login
    Login --> SystemCore
    
    %% 系统核心初始化各组件
    SystemCore --> BKT
    SystemCore --> RAG
    SystemCore --> KG
    SystemCore --> Selector
    SystemCore --> Evaluator
    SystemCore --> Visualizer
    
    %% 数据存储关系
    BKT --> DB
    SystemCore --> DB
    KG --> Cache
    KG --> Hash
    RAG --> VectorIndex
    
    %% RAG引擎相关连接
    RAG --> DB
    RAG --> BGE
    RAG --> Pangu
    RAG --> Cache
    BGE --> VectorIndex
    
    %% 知识图谱构建器相关连接
    KG --> DB
    KG --> Pangu
    KG --> Cache
    
    %% 题目选择器相关连接
    Selector --> RAG
    Selector --> BKT
    Selector --> Pangu
    Selector --> DB
    
    %% 评估器相关连接
    Evaluator --> Pangu
    Evaluator --> BKT
    Evaluator --> DB
    
    %% 可视化器相关连接
    Visualizer --> Cache
    Visualizer --> DB
    
    %% 硬件加速
    Pangu --> NPU
    BGE --> NPU
    
    style GradioUI fill:#E3F2FD
    style Login fill:#C8E6C9
    style SystemCore fill:#F3E5F5
    style Pangu fill:#FFF3E0
    style BGE fill:#FFF3E0
    style NPU fill:#E0E0E0
    style RAG fill:#E1BEE7
    style Selector fill:#E1BEE7
    style Evaluator fill:#E1BEE7
    style Visualizer fill:#E1BEE7
    style BKT fill:#C8E6C9
    style KG fill:#BBDEFB
    style DB fill:#FFE0B2
    style Cache fill:#FFE0B2
    style Hash fill:#FFE0B2
    style VectorIndex fill:#FFE0B2
```

### 主要技术栈

```mermaid
%%{init: {'theme':'base', 'themeVariables': {'primaryBackgroundColor':'#ffffff', 'primaryTextColor':'#000000', 'primaryBorderColor':'#333333', 'lineColor':'#333333', 'edgeLabelBackground':'#ffffff', 'secondaryColor':'#f0f0f0', 'tertiaryColor':'#ffffff', 'mainBkgColor':'#ffffff', 'secondBkgColor':'#f5f5f5', 'tertiaryBkgColor':'#ffffff', 'background':'#ffffff', 'primaryColor':'#ffffff'}}}%%
graph LR
    subgraph 前端技术["前端技术"]
        Gradio["Gradio 5.49.1<br/>Apache 2.0"]
    end
    
    subgraph 深度学习框架["深度学习框架"]
        PyTorch["PyTorch 2.5.1<br/>BSD 3-Clause"]
        TorchNPU["torch-npu 2.5.1<br/>Apache 2.0"]
    end
    
    subgraph 模型库["模型库"]
        Transformers["Transformers 4.53.2<br/>Apache 2.0"]
        VLLM["vLLM 0.9.2<br/>Apache 2.0"]
        VLLMAscend["vllm-ascend 0.9.2rc1"]
    end
    
    subgraph 数据处理["数据处理"]
        NumPy["NumPy 1.26.4<br/>BSD 3-Clause"]
        Pandas["Pandas 1.5.3<br/>BSD 3-Clause"]
        SciPy["SciPy 1.15.3<br/>BSD 3-Clause"]
    end
    
    subgraph 图论可视化["图论与可视化"]
        NetworkX["NetworkX 3.5<br/>BSD 3-Clause"]
        Plotly["Plotly 6.4.0<br/>MIT"]
        Matplotlib["Matplotlib<br/>Matplotlib License"]
    end
    
    subgraph AI模型["AI模型"]
        OpenPangu["OpenPangu 7B<br/>OpenPangu License v1.0"]
        BGE["BGE-small-zh-v1.5<br/>Apache 2.0"]
    end
    
    subgraph 硬件加速["硬件加速"]
        NPU["昇腾910B2 NPU"]
    end
    
    Gradio --> PyTorch
    Gradio --> Transformers
    PyTorch --> TorchNPU
    TorchNPU --> NPU
    Transformers --> OpenPangu
    Transformers --> BGE
    VLLM --> VLLMAscend
    VLLMAscend --> NPU
    OpenPangu --> NPU
    
    NumPy --> PyTorch
    Pandas --> NumPy
    SciPy --> NumPy
    NetworkX --> NumPy
    Plotly --> NumPy
    
    style Gradio fill:#BBDEFB
    style PyTorch fill:#C8E6C9
    style Transformers fill:#FFF9C4
    style OpenPangu fill:#FFE0B2
    style BGE fill:#FFE0B2
    style NPU fill:#E0E0E0
```

### 核心算法流程

```mermaid
%%{init: {'theme':'base', 'themeVariables': {'primaryBackgroundColor':'#ffffff', 'primaryTextColor':'#000000', 'primaryBorderColor':'#333333', 'lineColor':'#333333', 'edgeLabelBackground':'#ffffff', 'secondaryColor':'#f0f0f0', 'tertiaryColor':'#ffffff', 'mainBkgColor':'#ffffff', 'secondBkgColor':'#f5f5f5', 'tertiaryBkgColor':'#ffffff', 'background':'#ffffff', 'primaryColor':'#ffffff'}}}%%
flowchart TD
    Start([用户登录系统]) --> Auth{身份验证}
    Auth -->|学生| StudentFlow[学生流程]
    Auth -->|教师| TeacherFlow[教师流程]
    
    StudentFlow --> BKT1[【BKT算法】<br/>从数据库读取学生掌握度]
    BKT1 --> RAG[【知识图谱RAG引擎】<br/>基于知识图谱检索相关题目]
    RAG --> EntityExtract[【知识图谱】<br/>分析题目和知识点关系]
    EntityExtract --> Selector[【题目选择器】<br/>结合BKT+RAG+知识图谱<br/>智能推荐候选题目]
    Selector --> QuestionSelect[【盘古：题目互选】<br/>从候选题目中选择<br/>一道最合适的题目]
    QuestionSelect --> Answer[学生答题]
    Answer --> AnswerJudge[【盘古：答案评判】<br/>评判答案正误]
    AnswerJudge --> SaveDB[【数据库】<br/>保存答题记录]
    SaveDB --> BKT2[【BKT算法】<br/>更新学生掌握度到数据库]
    BKT2 --> Decision{还有题目?}
    Decision -->|是| Next[继续下一题]
    Decision -->|否| Analysis[【盘古：分析总结】<br/>分析学习情况<br/>生成总结报告]
    Next --> BKT1
    Analysis --> Report[生成学习报告]
    Report --> End([结束])
    
    TeacherFlow --> Manage[【教师管理功能】<br/>题库管理、学生管理、数据查看]
    
    style Start fill:#E8F5E9
    style Auth fill:#FFF3E0
    style StudentFlow fill:#E3F2FD
    style TeacherFlow fill:#F3E5F5
    style BKT1 fill:#E3F2FD
    style RAG fill:#F3E5F5
    style EntityExtract fill:#FFF3E0
    style Selector fill:#E1BEE7
    style QuestionSelect fill:#FFE0B2
    style AnswerJudge fill:#FFE0B2
    style SaveDB fill:#C8E6C9
    style BKT2 fill:#E3F2FD
    style Analysis fill:#FFE0B2
    style Report fill:#C8E6C9
    style End fill:#FFCDD2
    style Manage fill:#E1BEE7
```

### 系统架构

```mermaid
%%{init: {'theme':'base', 'themeVariables': {'primaryBackgroundColor':'#ffffff', 'primaryTextColor':'#000000', 'primaryBorderColor':'#333333', 'lineColor':'#333333', 'edgeLabelBackground':'#ffffff', 'secondaryColor':'#f0f0f0', 'tertiaryColor':'#ffffff', 'mainBkgColor':'#ffffff', 'secondBkgColor':'#f5f5f5', 'tertiaryBkgColor':'#ffffff', 'background':'#ffffff', 'primaryColor':'#ffffff'}}}%%
graph TB
    subgraph 前端层["前端层"]
        UI[Gradio UI界面<br/>enhanced_main_ui.py]
        Login[登录注册系统]
    end
    
    subgraph 业务逻辑层["业务逻辑层"]
        Core[系统核心<br/>system_core_db.py]
        BKT[BKT算法适配器<br/>bkt_database_adapter.py]
        RAG[RAG引擎<br/>rag_engine.py]
        KG[知识图谱构建器<br/>kg_builder.py]
    end
    
    subgraph 数据层["数据层"]
        DB[(SQLite数据库<br/>database.py)]
        Cache[知识图谱缓存<br/>knowledge_graph.pkl]
    end
    
    subgraph 模型层["模型层"]
        LLM[盘古7B模型<br/>llm_models.py]
        Embed[BGE嵌入模型<br/>embedding_model.py]
    end
    
    UI --> Login
    Login --> Core
    Core --> BKT
    Core --> RAG
    Core --> KG
    BKT --> DB
    RAG --> KG
    RAG --> Embed
    KG --> LLM
    KG --> Cache
    Core --> DB
    Core --> LLM
    Core --> Embed
    
    style UI fill:#BBDEFB
    style Login fill:#C8E6C9
    style Core fill:#FFF9C4
    style DB fill:#FFE0B2
    style LLM fill:#E1BEE7
    style Embed fill:#E1BEE7
```

## 💻 系统要求

### 硬件要求


#### 环境配置（参考）
- **CPU**: Kunpeng-920处理器
- **存储**: 196GB总容量（建议至少100GB可用空间用于模型文件和Docker镜像）
- **NPU**: 昇腾910B2 NPU

### 软件要求

- **操作系统**: Linux (推荐 Ubuntu 22.04.5 LTS，容器内使用 Ubuntu 22.04.5 LTS)
- **Python**: 3.11.12 (容器内版本)
- **Docker**: 18.09.0+ (宿主机版本，推荐使用Docker容器化部署)
- **昇腾CANN**: 23.0.6 (如果使用NPU，需要在宿主机安装，容器内驱动版本 23.0.6)

### 部署环境

#### 创建容器

```bash
docker run -dit --net=host \
  --name docker_person_exam \
  --device /dev/davinci6 \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /opt/pangu:/opt/pangu \
  -v /home:/home \
  quay.io/ascend/cann:pangu-8.1.rc1-910b-py3.11
```

### 2. 复制代码到容器内部

代码需要先上传到SSH服务器，然后复制到容器内。

#### 步骤1: 上传代码到SSH服务器

使用 `你喜欢的工具` 将代码上传到远程服务器：

#### 步骤2: 复制代码到容器内

在SSH服务器上，将代码复制到容器的 `/app` 目录（标准应用目录）：

```bash
# 复制代码到容器内
docker cp /home/xxx/PersonalExam/. docker_person_exam:/app/ # xxx为你上传的目录
```
![显示](img/8ebce37f131e58737321f7b3559c7c81.png)
**注意**：
- 容器内代码路径: `/app/education/`
- 数据目录: `/app/education/data/`（包含数据库文件 `education_system.db`，建议通过volume挂载持久化）
- 日志目录: `/app/education/logs/`（可通过volume挂载持久化）
- 数据库文件: `/app/education/data/education_system.db`（SQLite数据库，包含所有用户、题目和答题记录）

### 3. 下载bge-small-zh-v1.5模型

模型地址：https://huggingface.co/BAAI/bge-small-zh-v1.5

```bash
curl -LsSf https://hf.co/cli/install.sh | bash # Make sure the hf CLI is installed

# 方式1: 直接下载到项目目录（推荐）
# 下载到项目 models 目录，与配置文件中的默认路径一致
hf download BAAI/bge-small-zh-v1.5 --local-dir /app/education/models/bge-small-zh-v1.5

# 方式2: 使用默认位置下载
# 默认下载到: ~/.cache/huggingface/hub/
# 下载后需要配置 BGE_MODEL_PATH 环境变量指向实际路径
hf download BAAI/bge-small-zh-v1.5
```

### 4. 下载OpenPangu-Embedded-7B-V1.1模型

**重要提示**：OpenPangu模型采用 OPENPANGU MODEL LICENSE AGREEMENT VERSION 1.0 许可证，使用前请仔细阅读并遵守许可协议条款。

模型地址：https://huggingface.co/FreedomIntelligence/openPangu-Embedded-7B-V1.1

```bash
# 安装Hugging Face CLI（如果未安装）
curl -LsSf https://hf.co/cli/install.sh | bash

# 创建模型目录
mkdir -p /opt/pangu

# 下载模型到指定目录
hf download FreedomIntelligence/openPangu-Embedded-7B-V1.1 --local-dir /opt/pangu/openPangu-Embedded-7B-V1.1
```

**验证模型文件**：

```bash
# 检查模型文件是否存在
ls -lh /opt/pangu/openPangu-Embedded-7B-V1.1/

# 应该包含模型权重文件（如 .bin, .safetensors 等）和配置文件（config.json, tokenizer.json 等）
# 可进一步参考盘古的说明，检查checksum，确保文件没有损坏
```

**注意事项**：
- 模型文件较大（约14GB），请确保有足够的存储空间
- 下载可能需要较长时间，建议使用稳定的网络连接
- 模型路径必须与 `PANGU_MODEL_PATH` 环境变量配置一致（默认 `/opt/pangu/openPangu-Embedded-7B-V1.1`）

### 5. 准备题库数据

**重要**: 初次使用必须先准备题库数据，将JSON格式的题目文件转换为数据库格式。

#### 步骤1: 准备JSON题库文件

将JSON格式的题库文件保存在 `education/data/` 目录下。

题目JSON格式示例：

```json
[
  {
    "题号": 1,
    "问题": "解方程 x^2 - 5x + 6 = 0",
    "答案": "x = 2 或 x = 3",
    "解析": "使用因式分解法...",
    "难度": 0.3,
    "知识点大类": "代数",
    "知识点小类": "一元二次方程"
  }
]
```

#### 步骤2: 迁移数据到数据库

使用迁移脚本将JSON格式转换为数据库格式：

```bash
# 进入容器
docker exec -it docker_person_exam /bin/bash
cd /app/education

# 运行迁移脚本
python migrate_to_database.py
```

**注意**: 
- 系统首次启动时会自动创建默认用户账号：
  - 学生账号: `student_001` / `123456`
  - 教师账号: `teacher` / `admin123`
- 后续添加题目可以通过UI界面进行（见使用指南中的"题库管理"部分）

### 6. 启动系统

默认模型路径（通过环境变量配置）：
- 盘古7B模型: 通过 `PANGU_MODEL_PATH` 环境变量配置，默认 `/opt/pangu/openPangu-Embedded-7B-V1.1`
- BGE嵌入模型: 通过 `BGE_MODEL_PATH` 环境变量配置
  - **本地开发环境默认值**: `education/models/bge-small-zh-v1.5`
  - **Docker容器环境默认值**: `/app/education/models/bge-small-zh-v1.5`
  - 如果未设置环境变量，将使用上述默认路径

**启动步骤**:

```bash
# 进入运行中的容器
docker exec -it docker_person_exam /bin/bash
cd /app

# 安装依赖（使用国内镜像源，速度更快）
# 由于torch torch-npu torchaudio torchvision vllm vllm-ascend这几个包已经在docker中存在，安装是可能造成版本冲突，故不在requirements.txt中申明
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 在容器内启动系统
cd /app/education

# 设置环境变量（根据实际路径调整）
export PANGU_MODEL_PATH=/opt/pangu/openPangu-Embedded-7B-V1.1

python main.py
```

**重要提示**:
- 初次运行构建知识图谱需要一定时间，请耐心等待
- 服务将会运行在: http://localhost:7860（默认端口）
- 如果要修改端口，请在 `education/config.py` 中修改 `UI_CONFIG['port']` 的值，或通过环境变量 `GRADIO_SERVER_PORT` 设置
- 确保机器已经开放相应端口，如果是云服务器还需要在安全组中设置

![启动界面](img/416d63f2ace0064e16c1de10565bd9f6.png)

![运行界面](img/3a017ea219588d140d900805130dd2cd.png)

## ⚙️ 配置说明

### 主要配置项

配置文件: `education/config.py`

```python
# 模型配置
PANGU_MODEL_CONFIG = {
    "max_new_tokens": 32768,
    "temperature": 0.7,
    "top_p": 0.9,
    "device": "npu",
}

# 智能出题配置
SMART_QUESTION_CONFIG = {
    "default_question_count": 10,
    "weak_threshold": 0.4,  # 薄弱点阈值
    "weak_point_focus_ratio": 0.7,  # 薄弱点题目比例
    "rebuild_kg": False,  # 是否强制重建知识图谱
    "use_kg_rag": True,  # 是否使用知识图谱RAG
    "kg_rag_top_k": 5,  # 知识图谱RAG返回的候选题目数量
}

# UI配置
UI_CONFIG = {
    "port": 7860,
    "share": False,
    "server_name": "0.0.0.0"
}

# 日志配置
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_file": "logs/system.log"
}
```

### 数据库配置

数据库文件路径在 `main.py` 中配置，默认路径为：
- **数据库文件**: `education/data/education_system.db`
- **知识图谱缓存**: `education/data/knowledge_graph.pkl`
- **知识图谱哈希**: `education/data/kg_hash.txt`

数据库会自动创建，包含以下表：
- `users`: 用户表（学生和教师）
- `questions`: 题目表
- `student_states`: 学生状态表（BKT参数）
- `answer_records`: 答题记录表

### 环境变量

可以通过环境变量覆盖配置：

```bash
export PANGU_MODEL_PATH="/path/to/pangu/model"
export BGE_MODEL_PATH="/path/to/bge/model"
export GRADIO_SERVER_PORT=7860  # 修改UI端口
```

## 📖 使用指南

### 1. 访问系统

打开浏览器访问系统地址（默认 http://localhost:7860），系统启动后会显示登录界面。

![登录界面](img/image.png)

### 2. 用户注册与登录

#### 注册新用户（初次使用推荐）

1. 点击"注册新用户"按钮
2. 填写用户名、密码和角色（学生/教师）
3. 完成注册后即可使用新账号登录

![注册界面](img/image-2.png)

#### 使用默认账号登录

系统提供默认测试账号：
- **学生账号**: `student_001` / `123456`
- **教师账号**: `teacher` / `admin123`

![登录界面](img/微信图片_2025111211331860_1.jpg)

### 3. 学生功能

#### 开始智能测评

1. **登录学生账号**: 使用学生账号登录系统
2. **选择题目数量**: 在"智能测评"标签页选择题目数量（5-20题）
3. **开始测评**: 点击"开始智能测评"按钮
4. **答题**: 
   - 系统会根据学生掌握度智能推荐题目
   - 在答题框中输入答案
   - 点击"提交答案"进行判断
   - 查看答案反馈后，点击"下一题"继续
5. **完成测评**: 所有题目完成后，系统会生成个性化的评估报告

![答题界面](img/image-3.png)

![提交答案](img/image-4.png)

![测评界面](img/微信图片_2025111211331860_2.jpg)

![测评报告](img/微信图片_2025111211331860_3.jpg)

#### 查看学习分析

1. 切换到"我的学习"或"学习分析"标签页
2. 查看历史答题记录
3. 查看掌握度雷达图和学习统计

![学习分析](img/image-5.png)

![学习分析详情](img/59ced7bb989570f31c327b14d0abf9ff.png)

![雷达图](img/036e6c21995062bdcf82d63a537d9cca.png)

#### 查看知识图谱

1. 切换到"知识图谱"标签页
2. 选择布局算法（spring/circular/kamada_kawai）
3. 查看题目、知识点和难度之间的关系网络

![知识图谱](img/image-6.png)

![知识图谱详情](img/9d9937f6ba0c093f3ed85f806d972cd6.png)

### 4. 教师功能

#### 查看学生答题情况

1. **登录教师账号**: 使用教师账号登录系统
2. 在相应界面选择要查看的学生
3. 查看该学生的详细答题情况和学习数据

![学生管理](img/image-7.png)

#### 题库管理

1. 切换到"题库管理"标签页
2. 查看所有题目列表和统计信息
3. **添加新题目**: 点击"添加题目"按钮，填写题目信息（问题、答案、解析、难度、知识点等）即可添加到题库中

![题库管理](img/image-8.png)

![题库管理详情](img/0148140158105.png)
## 📁 项目结构

```
PersonalExam/
├── education/                    # 主程序目录
│   ├── main.py                  # 程序入口（数据库版本，带登录功能）
│   ├── config.py                # 配置文件
│   ├── database.py              # 数据库管理器（SQLite）
│   ├── system_core.py           # 系统核心（JSON版本，已弃用）
│   ├── system_core_db.py        # 系统核心（数据库版本）
│   ├── bkt_database_adapter.py  # BKT算法数据库适配器
│   ├── enhanced_main_ui.py      # 增强UI（带登录注册系统）
│   ├── migrate_to_database.py   # 数据迁移脚本（JSON→数据库）
│   ├── visualize_knowledge_graph.py  # 知识图谱可视化脚本
│   ├── data/                    # 数据目录
│   │   ├── education_system.db  # SQLite数据库（用户、题目、记录）
│   │   ├── knowledge_graph.pkl  # 知识图谱缓存
│   │   ├── kg_hash.txt          # 知识图谱哈希（用于检测更新）
│   │   ├── question_database.json      # 题库（旧版，用于迁移）
│   │   ├── question_database_2.json    # 题库（用于迁移）
│   │   └── student_states.json          # 学生状态（旧版，已迁移到数据库）
│   ├── models/                  # 模型模块
│   │   ├── llm_models.py        # 语言模型（盘古7B）
│   │   └── embedding_model.py   # 嵌入模型（BGE）
│   ├── data_management/         # 数据管理
│   │   └── question_db.py       # 题库数据库（旧版，已整合到database.py）
│   ├── knowledge_management/    # 知识管理
│   │   ├── kg_builder.py        # 知识图谱构建器
│   │   └── rag_engine.py        # RAG引擎（基于知识图谱）
│   ├── utils/                   # 工具模块
│   │   ├── bkt_algorithm.py     # BKT算法（基础实现）
│   │   ├── evaluator.py         # 评估器
│   │   └── question_generator.py # 题目生成器
│   ├── visualization/           # 可视化
│   │   ├── kg_visualizer.py     # 知识图谱可视化
│   │   └── radar_chart.py       # 雷达图可视化
│   ├── ui/                       # UI界面（旧版）
│   │   └── main_ui.py            # Gradio界面（已迁移到enhanced_main_ui.py）
│   └── logs/                     # 日志目录
│       └── system.log            # 系统日志
├── img/                          # 图片资源目录
│   └── *.png, *.jpg             # 文档和UI使用的图片
├── requirements.txt              # Python依赖
├── LICENSE                       # 项目许可证
├── NOTICE                        # 开源软件声明
└── README.md                     # 本文件
```

## 📄 开源许可证

### 项目代码许可证

**本项目代码采用 BSL 1.1 (Business Source License 1.1) 许可证**

BSL 1.1 是一种源代码可见的许可证，允许：
- ✅ **非商业使用**: 个人、教育、研究用途完全免费
- ✅ **查看源代码**: 可以查看、复制、修改源代码
- ✅ **分发**: 可以分发源代码（需保留版权声明）
- ⚠️ **商业使用限制**: 商业使用需要获得授权许可


完整的许可证文本请参阅 [LICENSE](LICENSE) 文件。

### OpenPangu模型许可证

本项目使用了OpenPangu模型（openPanGu-Embedded-7B-V1.1）。OpenPangu是由华为技术有限公司（Huawei Technologies Co., Ltd.）发布的大型语言模型。

**OpenPangu模型采用 OPENPANGU MODEL LICENSE AGREEMENT VERSION 1.0 许可证**

使用OpenPangu模型时，必须遵守OPENPANGU MODEL LICENSE AGREEMENT VERSION 1.0的所有条款和条件，包括：
- 地理限制：不能在欧盟境内使用
- 归属声明：必须包含OpenPangu的归属声明
- 许可证通知：必须包含许可证副本或链接

有关详细信息，请参阅：
- OpenPangu 7B模型 (Hugging Face): https://huggingface.co/FreedomIntelligence/openPangu-Embedded-7B-V1.1
- OpenPangu官方仓库: https://ai.gitcode.com/ascend-tribe/openpangu-embedded-7b-model

### 主要依赖第三方开源软件

本项目基于以下优秀的开源项目构建，我们感谢所有开源贡献者的工作。

（版本信息基于 requirements.txt）：

- **[Gradio](https://www.gradio.app/)** (5.49.1, Apache 2.0) - 用于构建交互式Web界面
- **[PyTorch](https://pytorch.org/)** (2.5.1, BSD 3-Clause) - 深度学习框架
- **[torch-npu](https://www.hiascend.com/)** (2.5.1.post1, Apache 2.0) - 昇腾NPU支持
- **[Transformers](https://huggingface.co/transformers)** (4.53.2, Apache 2.0) - Hugging Face 模型库
- **[vLLM](https://github.com/vllm-project/vllm)** (0.9.2, Apache 2.0) - 高性能LLM推理引擎
- **[vllm-ascend](https://www.hiascend.com/)** (0.9.2rc1, Apache 2.0) - vLLM昇腾支持
- **[NumPy](https://numpy.org/)** (1.26.4, BSD 3-Clause) - 数值计算库
- **[NetworkX](https://networkx.org/)** (3.5, BSD 3-Clause) - 图论和网络分析
- **[Plotly](https://plotly.com/python/)** (6.4.0, MIT) - 交互式数据可视化
- **[FastAPI](https://fastapi.tiangolo.com/)** (0.117.1, MIT) - 高性能Web API框架
- **[Accelerate](https://github.com/huggingface/accelerate)** (1.10.1, Apache 2.0) - 模型加速库

### 第三方依赖许可证

本项目使用的第三方依赖许可证类型主要是**商业友好**的：

完整的开源软件清单和许可证信息请参考：

- 📋 [LICENSE](LICENSE) - 项目代码的许可证
- 📄 [NOTICE](NOTICE) - 开源软件声明文件
- 📦 [requirements.txt](requirements.txt) - Python 依赖清单

### AI 模型

本项目使用的 AI 模型：

- **OpenPangu模型** (openPanGu-Embedded-7B-V1.1)
  - 来源: 华为技术有限公司 (Huawei Technologies Co., Ltd.)
  - 许可证: **OPENPANGU MODEL LICENSE AGREEMENT VERSION 1.0**
  - 参考: https://huggingface.co/FreedomIntelligence/openPangu-Embedded-7B-V1.1
  - **重要**: 本项目使用OpenPangu模型，需遵守OPENPANGU MODEL LICENSE AGREEMENT VERSION 1.0许可协议

- **BGE-small-zh-v1.5**
  - 来源: 北京智源人工智能研究院 (BAAI)
  - 许可证: Apache 2.0
  - 参考: https://github.com/FlagOpen/FlagEmbedding

### 使用说明

**重要**: 使用本项目时，必须遵守以下许可证要求：

1. **项目代码许可证 (BSL 1.1)**: 
   - 非商业使用：允许个人、教育、研究用途
   - 商业使用：需要获得授权许可

2. **OpenPangu模型许可证**: 必须遵守OPENPANGU MODEL LICENSE AGREEMENT VERSION 1.0的所有条款
   - 必须包含OpenPangu的归属声明
   - 必须包含许可证通知
   - 遵守OpenPangu商标使用规定
   - 不能在欧盟境内使用

3. **第三方开源软件**: 请遵守所有相关开源软件的许可证条款

详细信息请参考：
- 项目许可证: [LICENSE](LICENSE)
- 开源软件声明: [NOTICE](NOTICE)
- 各项目的官方文档

### 致谢

特别感谢：
- **华为技术有限公司** - 提供OpenPangu模型和昇腾计算资源支持
- **北京智源人工智能研究院 (BAAI)** - 提供BGE嵌入模型
- **Hugging Face** - 提供Transformers库和模型平台

### 引用和致谢

**重要声明**：如果您的研究、工作或项目中使用了本项目，请引用并致谢 **深圳市大数据研究院 (SRIBD)** 和 **AI系统与应用课题组**。

#### 引用格式

如果您在研究中使用了本项目，请使用以下格式引用：

**中文引用格式**：
```
基于LLM和知识图谱协同的个性化出题系统. 
AI系统与应用课题组, 深圳市大数据研究院 (SRIBD), 2025.
https://github.com/sribdcn/PersonalExam
```

**英文引用格式**：
```
Personalized Question Generation System Based on LLM and Knowledge Graph Collaboration.
AI System and Application Research Group, Shenzhen Research Institute of Big Data (SRIBD), 2025.
https://github.com/sribdcn/PersonalExam
```

**BibTeX格式**：
```bibtex
@software{personalexam2025,
  title = {基于LLM和知识图谱协同的个性化出题系统},
  author = {AI系统与应用课题组},
  organization = {深圳市大数据研究院 (SRIBD)},
  year = {2025},
  url = {https://github.com/sribdcn/PersonalExam}
}
```
```bibtex
@software{personalexam2025,
  title = {Personalized Question Generation System Based on LLM and Knowledge Graph Collaboration},
  author = {AI System and Application Research Group},
  organization = {Shenzhen Research Institute of Big Data (SRIBD)},
  year = {2025},
  url = {https://github.com/sribdcn/PersonalExam}
}
```

#### 致谢示例

在您的论文、报告或项目中，请包含以下致谢内容：

**中文致谢**：
```
本研究/工作使用了"基于LLM和知识图谱协同的个性化出题系统"。
感谢深圳市大数据研究院 (SRIBD) 和 AI系统与应用课题组提供该项目。
```

**英文致谢**：
```
This research/work uses the "Personalized Question Generation System Based on LLM and Knowledge Graph Collaboration".
We thank the Shenzhen Research Institute of Big Data (SRIBD) and the AI System and Application Research Group for providing this project.
```

## 🤝 贡献指南

我们欢迎所有形式的贡献！

### 如何贡献

1. **Fork** 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 **Pull Request**

### 代码规范

- 遵循 PEP 8 Python 代码规范
- 添加适当的注释和文档字符串
- 确保代码通过所有测试

### 报告问题

如果发现问题，请在 [Issues](https://github.com/sribdcn/PersonalExam /issues) 中报告，包括：
- 问题描述
- 复现步骤
- 预期行为
- 实际行为
- 环境信息

## ❓ 常见问题

### Q: 必须使用Docker吗？

A: 是的，本项目使用Docker容器化部署。Docker提供了更好的环境隔离和依赖管理，无需手动配置Python虚拟环境。


### Q: 如何添加新的题目？

A: 根据使用场景选择不同方式：

1. **初次使用（必须）**: 先将JSON格式的题库文件保存在 `education/data/` 目录下，然后运行迁移脚本：
   ```bash
   python migrate_to_database.py
   ```
   将JSON格式转换为数据库格式。

2. **后续添加题目（推荐）**: 使用教师账号登录，在"题库管理"标签页点击"添加题目"按钮，填写题目信息即可添加到题库中。

**注意**: 
- 初次使用必须先通过迁移脚本导入数据
- 系统会自动检测题库更新，并在需要时重建知识图谱
- 初次构建知识图谱需要一定时间，请耐心等待

### Q: 支持哪些知识点？

A: 系统支持自定义知识点结构。在 `education/config.py` 中的 `KNOWLEDGE_HIERARCHY` 配置知识点层级。可以通过volume挂载配置文件来修改。

### Q: 如何修改模型路径？

A: 有三种方式：
1. 在Dockerfile中配置（构建时）
2. 通过环境变量配置（运行时）
3. 挂载配置文件（运行时）

### Q: 系统支持多用户吗？

A: 是的，系统支持完整的用户管理系统：
- **用户注册**: 支持学生和教师两种角色的注册
- **用户登录**: 每个用户有独立的账号和密码
- **数据隔离**: 每个学生有独立的学习档案和状态
- **数据持久化**: 所有数据存储在SQLite数据库中，容器重启后数据不会丢失
- **默认账号**: 系统提供默认测试账号（学生: `student_001` / `123456`，教师: `teacher` / `admin123`）

### Q: 如何导出学习报告？

A: 目前报告在Web界面显示。可以复制报告内容或使用浏览器的打印功能保存为PDF。

### Q: 容器启动失败怎么办？

A: 检查以下几点：
1. 查看容器日志：`docker logs personal-exam`
2. 确认端口7860未被占用（本地部署）
3. 确认NPU设备正确挂载（如果使用NPU）
4. 确认模型文件路径正确
5. 远程部署时，确认所有volume挂载路径存在且可访问

### Q: 如何进行远程SSH部署？

A: 远程SSH部署步骤：
1. 使用SSH密钥连接到远程服务器：`ssh -i .cursor/liqinsi_key liqinsi@1.95.160.102`
2. 上传代码到服务器的 `/home/liqinsi/Documents/project/PersonalExam` 目录
3. 创建容器（参考"创建容器"章节，容器名: `docker_person_exam`）
4. 复制代码到容器内：`docker cp /home/liqinsi/Documents/project/PersonalExam/. docker_person_exam:/app/`
5. 进入容器：`docker exec -it docker_person_exam /bin/bash`
6. 安装依赖：`cd /app && pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`
7. 在容器内运行：`cd /app/education && python main.py`

**注意**: 数据库文件存储在 `education/data/education_system.db`，建议通过volume挂载持久化。

## 📝 更新日志

### Version 2.0.0 (当前版本)

- ✨ **数据库架构**: 从JSON文件存储迁移到SQLite数据库存储
- ✨ **用户系统**: 新增用户登录注册功能，支持学生和教师角色分离
- ✨ **知识图谱**: 新增知识图谱自动构建和RAG引擎
- ✨ **自动更新**: 知识图谱支持自动检测题库更新并重建
- ✨ **数据迁移**: 提供从JSON格式到数据库的迁移工具
- ✨ **增强UI**: 全新的UI界面，支持登录注册和角色权限管理
- ✨ 细粒度知识点追踪（支持知识点小类）
- ✨ RAG引擎驱动的智能题目推荐
- ✨ 知识图谱可视化功能
- ✨ 盘古7B模型支持
- ✨ 题目选择策略
- 📚 完善文档和许可证信息

### Version 1.0.1

- ✨ 新增细粒度知识点追踪（支持知识点小类）
- ✨ 新增RAG引擎驱动的智能题目推荐
- ✨ 新增知识图谱可视化功能
- ✨ 新增盘古7B模型支持
- ✨ 新增题目选择策略
- ✨ 新增学生状态保存
- 📚 完善文档和许可证信息

## 📚 相关资源

- [华为昇腾社区](https://www.hiascend.com/)
- [Hugging Face](https://huggingface.co/)
- [BAAI智源](https://www.baai.ac.cn/)
- [Gradio文档](https://www.gradio.app/docs/)

## 免责声明

由于 openPangu-Embedded-7B等AI模型固有的技术限制，以及本系统生成的内容是由盘古等模型自动生成的，我们无法对以下事项做出任何保证：
- 尽管本系统的输出由 AI 算法生成，但不能排除某些信息可能存在缺陷、不合理或引起不适的可能性，生成的内容不代表我们的态度或立场；
- 无法保证该系统 100% 准确、可靠、功能齐全、及时、安全、无错误、不间断、持续稳定或无任何故障；
- 该系统的输出内容不构成任何建议或决策，也不保证生成的内容的真实性、完整性、准确性、及时性、合法性、功能性或实用性。生成的内容不能替代教育、医疗等领域的专业人士回答您的问题。生成的内容仅供参考，不代表我们的任何态度、立场或观点。您需要根据实际情况做出独立判断，我们不承担任何责任。

## 反馈

如果有任何意见和建议，请提交issue或联系 aisa@sribd.cn。

**Made with ❤️ by [AI系统与应用课题组@SRIBD](aisa@sribd.cn)**

*最后更新: 2025年*



