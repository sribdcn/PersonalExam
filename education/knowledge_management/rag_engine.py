"""
LightRAG引擎模块
负责知识图谱的构建、查询和管理
"""

import asyncio
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from lightrag import LightRAG, QueryParam
    from lightrag.llm import openai_complete_if_cache, openai_embed
    LIGHTRAG_AVAILABLE = True
except ImportError:
    logger.warning("LightRAG未安装,将使用模拟模式")
    LIGHTRAG_AVAILABLE = False


class RAGEngine:
    def __init__(self, working_dir: str, embedding_func, llm_func=None):

        self.working_dir = Path(working_dir)  # 转换为Path对象便于操作
        self.working_dir.mkdir(exist_ok=True)  # 创建目录
        
        self.embedding_func = embedding_func  # 存储嵌入函数
        self.llm_func = llm_func  # 存储LLM函数
        self.rag = None  # LightRAG实例，初始为None
        
        logger.info(f"初始化RAG引擎,工作目录: {working_dir}")
    
    async def initialize(self):
        if not LIGHTRAG_AVAILABLE:
            logger.warning("LightRAG不可用,使用模拟模式")
            return
        
        try:
            async def simple_llm_func(prompt, **kwargs):
                return f"已处理: {prompt[:50]}..."
            
            self.rag = LightRAG(   # 创建LightRAG实例
                working_dir=str(self.working_dir),
                embedding_func=self.embedding_func,  # 嵌入函数
                llm_model_func=simple_llm_func   # 模拟LLM函数
            )
            await self.rag.initialize_storages()  # 初始化存储系统
            logger.info("RAG引擎初始化成功")
            
        except Exception as e:
            logger.error(f"RAG引擎初始化失败: {e}")
            raise
    
    async def insert_documents(self, documents: List[str]):   # 插入文档到RAG系统
        if self.rag is None:
            await self.initialize()
        
        if not LIGHTRAG_AVAILABLE:
            logger.warning("LightRAG不可用,跳过文档插入")
            return   # 模拟模式下不执行插入
        
        try:
            logger.info(f"正在插入 {len(documents)} 个文档...")
            for doc in documents:   # 遍历所有文档
                await self.rag.insert(doc)   # 异步插入每个文档
            logger.info("文档插入完成")
        except Exception as e:
            logger.error(f"文档插入失败: {e}")
            raise
    
    async def query(self, question: str, mode: str = "hybrid") -> str:   # 查询RAG系统
        if self.rag is None:
            await self.initialize()
        
        if not LIGHTRAG_AVAILABLE:
            logger.warning("LightRAG不可用,返回模拟结果")
            return f"模拟查询结果: {question}"
        
        try:
            result = await self.rag.query(
                question,   # 查询问题
                param=QueryParam(mode=mode)  # 查询参数
            )
            return result
        except Exception as e:
            logger.error(f"查询失败: {e}")
            return f"查询失败: {str(e)}"
    
    def get_graph_data(self) -> Optional[Dict[str, Any]]:  # 获取知识图谱数据
        if not LIGHTRAG_AVAILABLE or self.rag is None:
            return None
        
        try:
            graph_file = self.working_dir / "graph_chunk_entity_relation.graphml"
            if graph_file.exists():
                logger.info("图谱数据文件存在")
                return {"nodes": [], "edges": []}
            return None
        except Exception as e:
            logger.error(f"获取图谱数据失败: {e}")
            return None
    
    async def finalize(self):  # 关闭RAG引擎，释放资源
        if self.rag is not None and LIGHTRAG_AVAILABLE:
            try:
                await self.rag.finalize_storages()
                logger.info("RAG引擎已关闭")
            except Exception as e:
                logger.error(f"关闭RAG引擎失败: {e}")


class QuestionRAGManager:
    
    def __init__(self, rag_engine: RAGEngine):
        self.rag_engine = rag_engine
        logger.info("初始化题库知识图谱管理器")
    
    async def build_kg_from_questions(self, questions: List[Dict[str, Any]]):   # 从题目列表构建知识图谱 每个题目成为一个独立的文档
        documents = []
        for q in questions:
            # 构建文档文本
            doc_text = f"""
题目 {q.get('题号', 'N/A')}:
问题: {q.get('问题', '')}
答案: {q.get('答案', '')}
解析: {q.get('解析', '')}
难度: {q.get('难度', '')}
知识点: {q.get('知识点', '')}
"""
            documents.append(doc_text.strip())
        
        # 批量插入到RAG系统
        await self.rag_engine.insert_documents(documents)
        logger.info(f"已构建 {len(documents)} 道题目的知识图谱")
    
    async def find_similar_questions(self, knowledge_point: str,   # 查找相似题目
                                     difficulty: str = None,
                                     count: int = 5) -> List[Dict[str, Any]]:
        query_text = f"找出关于{knowledge_point}"    # 构建查询文本
        if difficulty:
            query_text += f"难度为{difficulty}"
        query_text += "的题目"
        
        result = await self.rag_engine.query(query_text, mode="hybrid")

        logger.info(f"查询知识点'{knowledge_point}'的相似题目")
        return []


def create_rag_engine(config: Dict[str, Any], embedding_func) -> RAGEngine:
    working_dir = config.get("working_dir", "./rag_storage")
    return RAGEngine(working_dir, embedding_func)


async def test_rag_engine():
    from models.embedding_model import create_embedding_model
    from config import BGE_M3_MODEL_PATH, EMBEDDING_MODEL_CONFIG, LIGHTRAG_CONFIG
    
    # 1. 创建嵌入模型
    embedding_model = create_embedding_model(BGE_M3_MODEL_PATH, EMBEDDING_MODEL_CONFIG)
    
    # 创建嵌入函数
    def embedding_func(texts):
        # 将文本列表转换为向量列表
        return embedding_model.encode(texts).tolist()
    
    # 创建RAG引擎
    rag_engine = create_rag_engine(LIGHTRAG_CONFIG, embedding_func)
    
    # 初始化
    await rag_engine.initialize()
    
    # 插入测试文档
    test_docs = [
        "题目1: 求解方程 x^2 - 5x + 6 = 0. 答案: x = 2 或 x = 3",
        "题目2: 计算积分 ∫x^2 dx. 答案: x^3/3 + C"
    ]
    await rag_engine.insert_documents(test_docs)
    # 查询
    result = await rag_engine.query("如何求解二次方程?")
    print(f"查询结果: {result}")

    await rag_engine.finalize()


if __name__ == "__main__":
    import sys
    sys.path.append("..")
    
    logging.basicConfig(level=logging.INFO)
    
    # 运行测试
    asyncio.run(test_rag_engine())
