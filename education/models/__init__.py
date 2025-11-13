# -*- coding: utf-8 -*-
"""
Copyright (c) 2025 AI系统与应用课题组@SRIBD

基于LLM和知识图谱协同的个性化出题系统 (PersonalExam)
Personalized Question Generation System Based on LLM and Knowledge Graph Collaboration

模型模块初始化
"""

from .llm_models import create_llm_model, PanGuModel
from .embedding_model import create_embedding_model

__all__ = [
    'create_llm_model',
    'PanGuModel',
    'create_embedding_model',
]