# -*- coding: utf-8 -*-
"""
Copyright (c) 2025 AI系统与应用课题组@SRIBD

基于LLM和知识图谱协同的个性化出题系统 (PersonalExam)
Personalized Question Generation System Based on LLM and Knowledge Graph Collaboration

学生掌握度雷达图可视化模块
结合BKT算法数据生成多种雷达图
"""

import logging
from typing import Dict, Any, Optional

import numpy as np
import plotly.graph_objects as go

logger = logging.getLogger(__name__)


class StudentRadarChart:
    """学生掌握度雷达图生成器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info("✅ 学生雷达图生成器初始化完成")

    def _ensure_categories(self, categories, values):
        """保证分类与数值长度匹配，且不为空"""
        if not categories or not values:
            return ["暂无数据"] * 4, [0.3] * 4
        if len(categories) != len(values):
            n = min(len(categories), len(values))
            categories = categories[:n]
            values = values[:n]
        return categories, values

    def create_radar_chart(self, student_profile: Dict[str, Any]) -> go.Figure:
        """
        按“知识点大类”创建雷达图
        """
        knowledge_points = student_profile.get("knowledge_points", {})

        categories = []
        mastery_values = []
        for major, minors in knowledge_points.items():
            mastery_list = []
            for minor_data in minors.values():
                if isinstance(minor_data, dict):
                    mastery_list.append(float(minor_data.get("mastery", 0.0)))
                elif isinstance(minor_data, (int, float)):
                    mastery_list.append(float(minor_data))
            if mastery_list:
                categories.append(str(major))
                mastery_values.append(float(np.mean(mastery_list)))

        categories, mastery_values = self._ensure_categories(categories, mastery_values)
        return self._build_figure(categories, mastery_values, title=f"学生掌握度雷达图 - {student_profile.get('student_id','')}")

    def create_detailed_radar_chart(self, student_profile: Dict[str, Any]) -> go.Figure:
        """
        按“知识点小类”创建雷达图
        """
        knowledge_points = student_profile.get("knowledge_points", {})

        categories = []
        mastery_values = []
        for major, minors in knowledge_points.items():
            for minor, data in minors.items():
                val = data.get("mastery", 0.0) if isinstance(data, dict) else float(data)
                categories.append(f"{major}/{minor}")
                mastery_values.append(float(val))

        categories, mastery_values = self._ensure_categories(categories, mastery_values)
        return self._build_figure(categories, mastery_values, title=f"详细掌握度雷达图 - {student_profile.get('student_id','')}", height=700)

    def create_knowledge_subgraph_radar(self,
                                        student_profile: Dict[str, Any],
                                        knowledge_subgraph: Dict[str, Any]) -> go.Figure:
        """
        结合知识子图实体生成雷达图（若没有实体则退回到小类雷达图）
        """
        entities = knowledge_subgraph.get("entities", [])
        knowledge_points = student_profile.get("knowledge_points", {})

        # 构建 mastery 映射
        mastery_map = {}
        for major, minors in knowledge_points.items():
            for minor, data in minors.items():
                mastery_map[f"{major}/{minor}"] = float(data.get("mastery", 0.0) if isinstance(data, dict) else data)

        categories = []
        mastery_values = []
        for entity in entities[:8]:  # 最多8个
            name = entity.get("name", "")
            matched = None
            for key in mastery_map.keys():
                if name in key or key in name:
                    matched = key
                    break
            categories.append(matched or name or "未知")
            mastery_values.append(mastery_map.get(matched, 0.3))

        if not categories:
            # 回退
            return self.create_detailed_radar_chart(student_profile)

        categories, mastery_values = self._ensure_categories(categories, mastery_values)
        return self._build_figure(categories, mastery_values, title=f"知识子图雷达图 - {student_profile.get('student_id','')}", height=600)

    def _build_figure(self, categories, mastery_values, title: str = "", height: int = 500) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=mastery_values,
            theta=categories,
            fill="toself",
            name="掌握度",
            line_color="rgb(32, 201, 151)",
            fillcolor="rgba(32, 201, 151, 0.3)",
            line_width=2
        ))
        fig.add_trace(go.Scatterpolar(
            r=[0.7] * len(categories),
            theta=categories,
            fill="none",
            name="目标线(70%)",
            line_color="rgb(255, 99, 71)",
            line_dash="dash",
            line_width=1
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], tickmode="linear", tick0=0, dtick=0.2),
                angularaxis=dict(rotation=90)
            ),
            showlegend=True,
            title=dict(text=title, x=0.5),
            height=height,
            template="plotly_white"
        )
        return fig


def create_radar_chart_generator(config: Optional[Dict[str, Any]] = None) -> StudentRadarChart:
    return StudentRadarChart(config)


