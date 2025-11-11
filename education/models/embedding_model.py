# -*- coding: utf-8 -*-
"""
Copyright (c) 2025 AI系统及应用课题组@SRIBD

基于LLM和知识图谱协同的个性化出题系统 (PersonalExam)
Personalized Question Generation System Based on LLM and Knowledge Graph Collaboration

嵌入模型接口模块
"""

import torch
import numpy as np
from typing import List, Union
from transformers import AutoModel, AutoTokenizer
import logging

logger = logging.getLogger(__name__)


class PanGuEmbedding:
    """盘古7B嵌入模型"""
    
    def __init__(self, model_path: str, config: dict):
        self.model_path = model_path
        self.config = config
        
        # 设备配置
        device_str = config.get("device", "cpu")
        if device_str == "npu":
            try:
                import torch_npu
                if torch.npu.is_available():
                    self.device = "npu:0"
                else:
                    self.device = "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device_str
            
        self.batch_size = config.get("batch_size", 32)
        self.max_length = config.get("max_length", 512)
        
        self.model = None
        self.tokenizer = None
        
        logger.info(f"初始化盘古7B嵌入模型: {model_path}")
    
    def load_model(self):
        """加载模型"""
        try:
            logger.info("正在加载盘古7B模型用于嵌入...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            self.model = AutoModel.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # 移动到设备
            if "npu" in self.device:
                import torch_npu
                self.model = self.model.to(self.device)
            else:
                self.model = self.model.to(self.device)
                
            self.model.eval()
            logger.info(f"盘古7B嵌入模型加载成功,设备: {self.device}")
            
        except Exception as e:
            logger.error(f"盘古7B嵌入模型加载失败: {e}")
            raise
    
    def encode(self, texts: Union[str, List[str]], 
               normalize: bool = True) -> np.ndarray:
        """编码文本"""
        if self.model is None:
            self.load_model()
        
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            encoded_input = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # 移动到设备
            if "npu" in self.device:
                encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
            else:
                encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
            
            with torch.no_grad():
                model_output = self.model(**encoded_input)
                # 使用[CLS] token的嵌入或平均池化
                batch_embeddings = model_output.last_hidden_state[:, 0, :].cpu().numpy()
            
            embeddings.append(batch_embeddings)
        
        embeddings = np.vstack(embeddings)
        
        if normalize:
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        return embeddings
    
    def get_embedding_dim(self) -> int:
        """获取嵌入维度"""
        if self.model is None:
            self.load_model()
        return self.model.config.hidden_size
    
    def __call__(self, texts: Union[str, List[str]]) -> np.ndarray:
        return self.encode(texts)


class BGE_M3_Embedding:
    
    def __init__(self, model_path: str, config: dict):
        self.model_path = model_path
        self.config = config
        self.device = config.get("device", "cpu")
        self.batch_size = config.get("batch_size", 32)
        self.max_length = config.get("max_length", 512)
        
        self.model = None
        self.tokenizer = None
        
        logger.info(f"初始化BGE嵌入模型: {model_path}")
    
    def load_model(self):
        try:
            logger.info("正在加载BGE模型...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModel.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"BGE模型加载成功,设备: {self.device}")
        except Exception as e:
            logger.error(f"BGE模型加载失败: {e}")
            raise
    
    def encode(self, texts: Union[str, List[str]], 
               normalize: bool = True) -> np.ndarray:
        if self.model is None:
            self.load_model()
        
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            encoded_input = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            ).to(self.device)
            
            with torch.no_grad():
                model_output = self.model(**encoded_input)
                batch_embeddings = model_output.last_hidden_state[:, 0, :].cpu().numpy()
            
            embeddings.append(batch_embeddings)
        
        embeddings = np.vstack(embeddings)
        
        if normalize:
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        return embeddings
    
    def get_embedding_dim(self) -> int:
        if self.model is None:
            self.load_model()
        return self.model.config.hidden_size
    
    def __call__(self, texts: Union[str, List[str]]) -> np.ndarray:
        return self.encode(texts)


class MockEmbedding:
    """模拟嵌入模型"""
    
    def __init__(self, model_path: str, config: dict):
        self.model_path = model_path
        self.config = config
        self.embedding_dim = 1024  # 盘古7B的嵌入维度
        logger.info(f"使用模拟嵌入模型 (路径不存在: {model_path})")
    
    def load_model(self):
        logger.warning("使用模拟嵌入模型,实际模型未加载")
    
    def encode(self, texts: Union[str, List[str]], 
               normalize: bool = True) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = np.random.randn(len(texts), self.embedding_dim)
        
        if normalize:
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        return embeddings
    
    def get_embedding_dim(self) -> int:
        return self.embedding_dim
    
    def __call__(self, texts: Union[str, List[str]]) -> np.ndarray:
        return self.encode(texts)


def create_embedding_model(model_path: str, config: dict):

    import os
    
    # 检查是否启用盘古嵌入
    use_pangu = config.get("use_pangu_embedding", True)
    pangu_path = config.get("pangu_model_path", "")
    
    if use_pangu and pangu_path:
        try:
            logger.info("尝试使用盘古7B的嵌入功能...")
            return PanGuEmbedding(pangu_path, config)
        except Exception as e:
            logger.warning(f"盘古7B嵌入初始化失败: {e}，尝试使用BGE")
    
    # 使用BGE-M3
    if os.path.exists(model_path):
        try:
            return BGE_M3_Embedding(model_path, config)
        except Exception as e:
            logger.warning(f"BGE加载失败: {e}，使用模拟模型")
            return MockEmbedding(model_path, config)
    else:
        logger.warning(f"嵌入模型路径不存在: {model_path}, 使用模拟模型")
        return MockEmbedding(model_path, config)


def lightrag_embedding_func(texts: Union[str, List[str]], 
                           embedding_model) -> List[List[float]]:

    embeddings = embedding_model.encode(texts)
    return embeddings.tolist()


if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from config import BGE_M3_MODEL_PATH, EMBEDDING_MODEL_CONFIG
    
    logging.basicConfig(level=logging.INFO)
    
    embedding_model = create_embedding_model(BGE_M3_MODEL_PATH, EMBEDDING_MODEL_CONFIG)
    embedding_model.load_model()
    
    test_texts = ["这是一个测试句子", "另一个测试"]
    embeddings = embedding_model.encode(test_texts)
    
    print(f"嵌入维度: {embedding_model.get_embedding_dim()}")
    print(f"嵌入形状: {embeddings.shape}")
    print(f"第一个向量: {embeddings[0][:5]}...")