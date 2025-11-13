# 开源软件许可证清单

本文档列出了本项目使用的所有开源软件（包括直接依赖和间接依赖）及其对应的开源协议。

**项目信息**:
- **开发单位**: 深圳市大数据研究院 (SRIBD)
- **课题组**: AI系统与应用课题组
- **年份**: 2025

**重要**: 本项目代码采用 **BSL 1.1 (Business Source License 1.1)** 许可证。

**项目代码许可证说明**:
- 许可证类型: BSL 1.1 (Business Source License 1.1)
- 允许使用: 非商业用途（个人、教育、研究）
- 商业使用: 需要获得授权许可
- 完整许可证文本: 请参阅项目根目录的 [LICENSE](LICENSE) 文件

**OpenPangu模型许可证**:
本项目使用的OpenPangu模型采用 OPENPANGU MODEL LICENSE AGREEMENT VERSION 1.0 许可证，需遵守该许可协议的所有条款和条件。

**最后更新**: 2025年（基于 requirements.txt 实际依赖）

## 📦 核心直接依赖（项目直接使用的库）

### 1. **Gradio** - UI框架
- **版本**: 5.49.1
- **用途**: 构建Web用户界面
- **开源协议**: **Apache 2.0**
- **官网**: https://www.gradio.app/
- **GitHub**: https://github.com/gradio-app/gradio

### 2. **PyTorch** - 深度学习框架
- **版本**: 2.5.1
- **用途**: 深度学习模型训练和推理
- **开源协议**: **BSD 3-Clause License**
- **官网**: https://pytorch.org/
- **GitHub**: https://github.com/pytorch/pytorch

### 3. **torch-npu** - 昇腾NPU支持
- **版本**: 2.5.1.post1
- **用途**: 在昇腾NPU上运行PyTorch模型
- **开源协议**: **Apache 2.0** (华为开源)
- **来源**: 华为昇腾社区

### 4. **Transformers** (Hugging Face) - 模型库
- **版本**: 4.53.2
- **用途**: 加载和使用预训练模型（OpenPangu、BGE等）
- **开源协议**: **Apache 2.0**
- **官网**: https://huggingface.co/
- **GitHub**: https://github.com/huggingface/transformers

### 5. **vLLM** - 高性能LLM推理
- **版本**: 0.9.2
- **用途**: 高性能大语言模型推理引擎
- **开源协议**: **Apache 2.0**
- **GitHub**: https://github.com/vllm-project/vllm

### 6. **vllm-ascend** - vLLM昇腾支持
- **版本**: 0.9.2rc1
- **用途**: vLLM在昇腾NPU上的支持
- **开源协议**: **Apache 2.0** (华为开源)
- **来源**: 华为昇腾社区

### 7. **NumPy** - 数值计算库
- **版本**: 1.26.4
- **用途**: 数组计算、向量运算
- **开源协议**: **BSD 3-Clause License**
- **官网**: https://numpy.org/
- **GitHub**: https://github.com/numpy/numpy

### 8. **NetworkX** - 图论库
- **版本**: 3.5
- **用途**: 知识图谱构建和操作
- **开源协议**: **BSD 3-Clause License**
- **官网**: https://networkx.org/
- **GitHub**: https://github.com/networkx/networkx

### 9. **Plotly** - 可视化库
- **版本**: 6.4.0
- **用途**: 知识图谱交互式可视化
- **开源协议**: **MIT License**
- **官网**: https://plotly.com/python/
- **GitHub**: https://github.com/plotly/plotly.py

### 10. **FastAPI** - Web框架
- **版本**: 0.117.1
- **用途**: 高性能Web API框架（Gradio依赖）
- **开源协议**: **MIT License**
- **官网**: https://fastapi.tiangolo.com/
- **GitHub**: https://github.com/tiangolo/fastapi

### 11. **Accelerate** - 模型加速
- **版本**: 1.10.1
- **用途**: Hugging Face模型加速库
- **开源协议**: **Apache 2.0**
- **GitHub**: https://github.com/huggingface/accelerate

### 12. **HuggingFace Hub** - 模型仓库
- **版本**: 0.35.1
- **用途**: 访问Hugging Face模型和数据集
- **开源协议**: **Apache 2.0**
- **GitHub**: https://github.com/huggingface/huggingface_hub

### 13. **Tokenizers** - 快速分词
- **版本**: 0.21.4
- **用途**: 高性能文本分词
- **开源协议**: **Apache 2.0**
- **GitHub**: https://github.com/huggingface/tokenizers

### 14. **Safetensors** - 安全张量格式
- **版本**: 0.6.2
- **用途**: 安全的模型权重存储格式
- **开源协议**: **Apache 2.0**
- **GitHub**: https://github.com/huggingface/safetensors

## 🤖 AI模型

### 15. **OpenPangu模型** (openPanGu-Embedded-7B-V1.1) ⚠️ **核心模型**
- **版本**: V1.1
- **用途**: 语言模型，用于答案评估、报告生成、题目选择
- **开源协议**: **OPENPANGU MODEL LICENSE AGREEMENT VERSION 1.0**
- **来源**: 华为技术有限公司 (Huawei Technologies Co., Ltd.)
- **路径**: `/opt/pangu/openPangu-Embedded-7B-V1.1`
- **参考**: https://ai.gitcode.com/ascend-tribe/openpangu-embedded-1b-model
- **重要说明**: 
  - 本项目基于OpenPangu模型构建
  - **必须遵守OPENPANGU MODEL LICENSE AGREEMENT VERSION 1.0的所有条款**
  - OpenPangu是华为技术有限公司的商标
  - 使用本项目的代码和模型时，需遵守该许可协议
  - 完整的许可证文本请参阅项目根目录的 [LICENSE](LICENSE) 文件

### 16. **BGE-small-zh-v1.5** - 嵌入模型
- **版本**: v1.5
- **用途**: 文本嵌入，用于RAG向量检索
- **开源协议**: **Apache 2.0** (BAAI开源)
- **来源**: 北京智源人工智能研究院 (BAAI)
- **路径**: 通过环境变量 `BGE_MODEL_PATH` 配置
- **说明**: BGE系列模型通常采用Apache 2.0协议

## 📋 完整依赖清单（按许可证分类）

### Apache 2.0 License
以下软件使用Apache 2.0许可证：

**核心框架**:
- Gradio (5.49.1)
- Transformers (4.53.2)
- torch-npu (2.5.1.post1)
- vLLM (0.9.2)
- Accelerate (1.10.1)
- HuggingFace Hub (0.35.1)
- Tokenizers (0.21.4)
- Safetensors (0.6.2)

**Web和网络**:
- FastAPI (0.117.1)
- FastAPI-CLI (0.0.13)
- FastAPI-Cloud-CLI (0.2.1)
- Starlette (0.48.0)
- Uvicorn (0.37.0)
- Hypercorn (0.17.3)
- HTTPX (0.27.2)
- HTTPCore (1.0.9)
- Aiofiles (24.1.0)
- Aiohttp (3.12.15)
- Python-multipart (0.0.20)
- Safehttpx (0.1.7)

**数据处理**:
- Datasets (3.6.0)
- Sentence-transformers (5.1.1)
- Sentencepiece (0.2.1)
- Regex (2025.9.18)
- Requests (2.32.3)
- Protobuf (3.20.0)
- PyArrow (21.0.0)
- H5py (3.14.0)

**工具库**:
- Absl-py (2.3.0)
- Addict (2.4.0)
- Fire (0.7.1)
- Ruff (0.14.4)
- Tiktoken (0.11.0)
- Outlines (0.1.11)
- Outlines-core (0.1.26)
- Llguidance (0.7.30)
- Lm-format-enforcer (0.10.12)
- Compressed-tensors (0.10.2)
- GGUF (0.17.1)
- VLLM-ascend (0.9.2rc1) - Apache 2.0

**其他**:
- FFmpy (0.6.4)
- OpenCV-python-headless (4.11.0.86)
- Sentry-sdk (2.39.0)
- Prometheus-fastapi-instrumentator (7.1.0)
- Prometheus-client (0.23.1)

### BSD 3-Clause License
以下软件使用BSD 3-Clause许可证：

**核心库**:
- PyTorch (2.5.1)
- Torchaudio (2.5.1)
- Torchvision (0.20.1)
- NumPy (1.26.4)
- NetworkX (3.5)
- SciPy (1.15.3)
- Pandas (1.5.3)
- Scikit-learn (1.5.0)

**Web框架**:
- Uvicorn (0.37.0)
- Websockets (15.0.1)
- MarkupSafe (3.0.3)
- Jinja2 (3.1.6)
- Werkzeug (3.1.3)
- Flask (3.1.2)
- Quart (0.20.0)
- Tornado (6.5.2)

**工具库**:
- FSSpec (2025.3.0)
- Sympy (1.13.1)
- Matplotlib (3.10.6)
- Seaborn (0.13.2)
- Pillow (11.3.0)
- Fonttools (4.60.0)
- Contourpy (1.3.3)
- Cycler (0.12.1)
- Kiwisolver (1.4.9)
- Pyparsing (3.2.5)

**其他**:
- Certifi (2025.4.26)
- Charset-normalizer (3.4.2)
- Idna (3.10)
- Urllib3 (2.4.0)
- Distro (1.9.0)
- Psutil (7.0.0)
- Threadpoolctl (3.6.0)

### MIT License
以下软件使用MIT许可证：

**核心库**:
- FastAPI (0.117.1)
- Pydantic (2.11.9)
- Pydantic-core (2.33.2)
- Pydantic-extra-types (2.10.5)
- Plotly (6.4.0)
- PyYAML (6.0.2)
- Pydub (0.25.1)
- Orjson (3.11.4)

**工具库**:
- Click (8.3.0)
- Rich (14.1.0)
- Rich-toolkit (0.15.1)
- Tqdm (4.67.1)
- Pygments (2.19.2)
- Tabulate (0.9.0)
- Prettytable (3.16.0)
- Termcolor (3.1.0)

**其他**:
- Python-dateutil (2.9.0.post0)
- Python-dotenv (1.1.1)
- Six (1.17.0)
- Zipp (3.23.0)
- Platformdirs (4.4.0)
- Typing-extensions (4.15.0)
- pyext (git) (https://github.com/refi64/PyExt.git)

### BSD 2-Clause License
- Decorator (5.2.1)

### Python Software Foundation License
- Typing-extensions (4.15.0)

### Public Domain / Unlicense
- Filelock (3.19.1)

### LGPL / GPL (需注意)
以下软件使用LGPL或GPL许可证，需要注意：

- **Cython** (3.1.1) - Apache 2.0 (主要), 部分Cython代码可能受其他许可证约束

### 其他许可证

**MPL 2.0 / MIT**:
- Tqdm (4.67.1) - 双重许可

**Apache 2.0 / MIT**:
- Orjson (3.11.4) - 双重许可

**需确认许可证**:
无

## 🔍 特殊依赖说明

### Git依赖
项目包含以下从Git仓库安装的依赖：

1. **pyext** (github.com/refi64/PyExt)
   - 用途: Python扩展工具
   - 许可证: **MIT License**

### 昇腾相关依赖
- **torch-npu** (2.5.1.post1) - Apache 2.0
- **vllm-ascend** (0.9.2rc1) - Apache 2.0
- 需要昇腾硬件和CANN工具包支持

## 📊 依赖统计

### 按许可证类型统计
- **Apache 2.0**: 约 40+ 个包
- **BSD 3-Clause**: 约 30+ 个包
- **MIT**: 约 20+ 个包
- **其他**: 约 10+ 个包

### 主要依赖层级
```
项目
├── Gradio (5.49.1, Apache 2.0)
│   ├── FastAPI (0.117.1, MIT)
│   ├── Uvicorn (0.37.0, BSD 3-Clause)
│   ├── Pydantic (2.11.9, MIT)
│   └── ... (更多依赖)
├── PyTorch (2.5.1, BSD 3-Clause)
│   ├── NumPy (1.26.4, BSD 3-Clause)
│   ├── NetworkX (3.5, BSD 3-Clause)
│   └── ... (更多依赖)
├── torch-npu (2.5.1.post1, Apache 2.0)
│   └── 需要昇腾CANN支持
├── Transformers (4.53.2, Apache 2.0)
│   ├── Tokenizers (0.21.4, Apache 2.0)
│   ├── Safetensors (0.6.2, Apache 2.0)
│   ├── Accelerate (1.10.1, Apache 2.0)
│   └── ... (更多依赖)
├── vLLM (0.9.2, Apache 2.0)
│   └── vllm-ascend (0.9.2rc1, Apache 2.0)
├── NetworkX (3.5, BSD 3-Clause)
│   ├── NumPy (1.26.4, BSD 3-Clause)
│   ├── SciPy (1.15.3, BSD 3-Clause)
│   └── ... (更多依赖)
└── Plotly (6.4.0, MIT)
    ├── NumPy (1.26.4, BSD 3-Clause)
    ├── Pandas (1.5.3, BSD 3-Clause)
    └── ... (更多依赖)
```

## 📚 Python 标准库

以下为Python标准库，无需单独声明许可证：
- `json`, `logging`, `pathlib`, `typing`, `dataclasses`
- `collections`, `datetime`, `re`, `random`, `threading`
- `difflib`, `sys`, `os`, `io`, `functools`, `itertools`
- 等等

## 📖 开源协议说明

### Apache 2.0 License
- ✅ 允许商业使用
- ✅ 允许修改
- ✅ 允许分发
- ✅ 允许专利使用
- ⚠️ 需要保留版权声明
- ⚠️ 需要包含许可证文件
- ⚠️ 修改文件需要标注

### BSD 3-Clause License
- ✅ 允许商业使用
- ✅ 允许修改
- ✅ 允许分发
- ⚠️ 需要保留版权声明
- ⚠️ 不能使用作者名义进行推广

### MIT License
- ✅ 允许商业使用
- ✅ 允许修改
- ✅ 允许分发
- ⚠️ 需要保留版权声明和许可证

## ⚠️ 重要注意事项

### 项目代码许可证

**本项目代码采用 BSL 1.1 (Business Source License 1.1) 许可证**

BSL 1.1 是一种源代码可见的许可证，主要特点：

**允许的使用**:
- ✅ 非商业使用（个人、教育、研究）
- ✅ 查看、复制、修改源代码
- ✅ 分发源代码（需保留版权声明）

**商业使用限制**:
- ⚠️ 商业使用需要获得授权许可
- ⚠️ 商业使用需要联系版权所有者获得许可

**OpenPangu模型许可证**:
本项目使用的OpenPangu模型采用 OPENPANGU MODEL LICENSE AGREEMENT VERSION 1.0 许可证，需遵守该许可协议的所有条款和条件，包括：
- ⚠️ 地理限制：不能在欧盟境内使用
- ⚠️ 必须包含OpenPangu的归属声明
- ⚠️ 必须包含许可证通知
- ⚠️ OpenPangu是华为技术有限公司的商标，使用时需遵守商标使用规定

### 第三方依赖许可证兼容性

本项目使用的第三方依赖许可证类型主要是**商业友好**的：

- ✅ **Apache 2.0**: 完全允许商业使用，专利友好
- ✅ **BSD 3-Clause**: 完全允许商业使用，限制最少
- ✅ **MIT**: 完全允许商业使用，限制最少

**无 GPL/LGPL 许可证**: 本项目未使用 GPL、AGPL、LGPL 许可证的依赖，因此：
- ✅ 可以闭源使用
- ✅ 可以商业使用
- ✅ 无需公开源代码


### 项目代码许可证

**本项目代码** 使用 **BSL 1.1 (Business Source License 1.1)** 许可证。

- 允许非商业使用，商业使用需要授权
- 完整的许可证文本请参阅项目根目录的 [LICENSE](LICENSE) 文件

### OpenPangu模型许可证

**OpenPangu模型 (openPanGu-Embedded-7B-V1.1)** 使用 **OPENPANGU MODEL LICENSE AGREEMENT VERSION 1.0**。

使用OpenPangu模型必须遵守该许可协议的所有条款和条件，包括地理限制、归属声明等要求。

## 📝 许可证合规建议

### ✅ 已完成的合规工作

1. **✅ 创建了 requirements.txt**
   - 包含所有依赖及其精确版本
   - 包含Git依赖的完整URL和commit hash

2. **✅ 创建了 NOTICE 文件**
   - 列出了所有使用的开源软件及其版权信息
   - 按许可证类型分类组织

3. **✅ 创建了许可证文档**
   - 详细的依赖清单和许可证信息
   - 包含直接依赖和间接依赖

### 📋 建议的后续步骤

#### 1. 确认Git依赖的许可证

所有Git依赖的许可证已确认：
- pyext: MIT License ✅

#### 2. 确认昇腾相关依赖的许可证

所有昇腾相关依赖的许可证已确认：
- vllm-ascend: Apache 2.0 ✅
- torch-npu: Apache 2.0 ✅

#### 3. 定期更新

- **每月检查**: 检查依赖库的更新和安全补丁
- **每季度审查**: 审查许可证信息是否有变化
- **版本发布前**: 更新所有许可证文档

## 📞 获取帮助

如有许可证相关问题：
- 查阅各项目的官方文档
- 参考 OSI 许可证说明: https://opensource.org/licenses
- 咨询法律顾问（商业项目）

## 参考资源

- [OSI开源许可证列表](https://opensource.org/licenses)
- [SPDX许可证标识符](https://spdx.org/licenses/)
- [华为昇腾社区](https://www.hiascend.com/)
- [Hugging Face](https://huggingface.co/)
- [BAAI智源](https://www.baai.ac.cn/)

---

**最后更新**: 2025年（基于 requirements.txt 实际依赖）
**维护者**: 项目开发团队
