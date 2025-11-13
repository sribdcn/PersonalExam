# -*- coding: utf-8 -*-
"""
Copyright (c) 2025 AI系统与应用课题组@SRIBD

基于LLM和知识图谱协同的个性化出题系统 (PersonalExam)
Personalized Question Generation System Based on LLM and Knowledge Graph Collaboration

知识图谱可视化模块
使用NetworkX和Plotly进行知识图谱的可视化
"""

import logging
import networkx as nx
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional, Tuple
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class KnowledgeGraphVisualizer:
    """知识图谱可视化器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化可视化器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.graph = nx.Graph()
        
        logger.info("知识图谱可视化器初始化完成")
    
    def build_graph_from_questions(self, questions: List[Dict[str, Any]]):
        """
        从题目数据构建知识图谱
        
        Args:
            questions: 题目列表
        """
        self.graph.clear()
        
        # 创建知识点节点
        knowledge_points = set()
        difficulty_levels = set()
        
        for q in questions:
            knowledge = q.get('知识点', '未知')
            difficulty = q.get('难度', '未知')
            question_id = q.get('题号', len(self.graph.nodes))
            
            knowledge_points.add(knowledge)
            difficulty_levels.add(difficulty)
            
            # 添加题目节点
            self.graph.add_node(
                f"Q{question_id}",
                type='question',
                title=q.get('问题', '')[:30] + '...',
                difficulty=difficulty,
                knowledge=knowledge,
                full_data=q
            )
            
            # 添加知识点节点(如果不存在)
            if not self.graph.has_node(knowledge):
                self.graph.add_node(
                    knowledge,
                    type='knowledge',
                    title=knowledge
                )
            
            # 添加难度节点(如果不存在)
            if not self.graph.has_node(difficulty):
                self.graph.add_node(
                    difficulty,
                    type='difficulty',
                    title=difficulty
                )
            
            # 添加边
            self.graph.add_edge(f"Q{question_id}", knowledge, relation='属于')
            self.graph.add_edge(f"Q{question_id}", difficulty, relation='难度为')
        
        # 添加知识点之间的关系
        for kp1 in knowledge_points:
            for kp2 in knowledge_points:
                if kp1 != kp2:
                    # 如果两个知识点有相似的题目,添加关联
                    q1 = [n for n, d in self.graph.nodes(data=True) 
                          if d.get('type') == 'question' and d.get('knowledge') == kp1]
                    q2 = [n for n, d in self.graph.nodes(data=True) 
                          if d.get('type') == 'question' and d.get('knowledge') == kp2]
                    
                    # 简单判断:如果题目数量相似,认为有关联
                    if len(q1) > 0 and len(q2) > 0:
                        similarity = min(len(q1), len(q2)) / max(len(q1), len(q2))
                        if similarity > 0.5:
                            if not self.graph.has_edge(kp1, kp2):
                                self.graph.add_edge(kp1, kp2, 
                                                  relation='相关',
                                                  weight=similarity)
        
        logger.info(f"知识图谱构建完成: {len(self.graph.nodes)} 个节点, "
                   f"{len(self.graph.edges)} 条边")
    
    def get_node_positions(self, layout: str = 'spring') -> Dict[str, Tuple[float, float]]:
        """
        获取节点位置
        
        Args:
            layout: 布局算法 (spring, circular, kamada_kawai)
            
        Returns:
            节点位置字典
        """
        if layout == 'spring':
            pos = nx.spring_layout(self.graph, k=2, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(self.graph)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(self.graph)
        else:
            pos = nx.spring_layout(self.graph)
        
        return pos
    
    def create_plotly_figure(self, layout: str = 'spring',
                           title: str = '知识图谱') -> go.Figure:
        """
        创建Plotly可视化图形
        
        Args:
            layout: 布局算法
            title: 图标题
            
        Returns:
            Plotly Figure对象
        """
        if len(self.graph.nodes) == 0:
            logger.warning("图谱为空,无法可视化")
            # 返回空图
            fig = go.Figure()
            fig.add_annotation(
                text="知识图谱为空<br>请先导入题目数据",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20)
            )
            return fig
        
        # 获取节点位置
        pos = self.get_node_positions(layout)
        
        # 创建边的轨迹
        edge_trace = []
        for edge in self.graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace.append(
                go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode='lines',
                    line=dict(width=1, color='#888'),
                    hoverinfo='none',
                    showlegend=False
                )
            )
        
        # 按类型分组节点
        node_traces = {}
        node_types = {
            'knowledge': {'color': '#FF6B6B', 'symbol': 'diamond', 'size': 30, 'name': '知识点'},
            'difficulty': {'color': '#4ECDC4', 'symbol': 'square', 'size': 25, 'name': '难度'},
            'question': {'color': '#95E1D3', 'symbol': 'circle', 'size': 15, 'name': '题目'}
        }
        
        for node_type, style in node_types.items():
            x_coords = []
            y_coords = []
            texts = []
            
            for node, data in self.graph.nodes(data=True):
                if data.get('type') == node_type:
                    x, y = pos[node]
                    x_coords.append(x)
                    y_coords.append(y)
                    texts.append(data.get('title', node))
            
            if x_coords:  # 只有当该类型有节点时才添加
                node_traces[node_type] = go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode='markers+text',
                    marker=dict(
                        size=style['size'],
                        color=style['color'],
                        symbol=style['symbol'],
                        line=dict(width=2, color='white')
                    ),
                    text=texts,
                    textposition='top center',
                    textfont=dict(size=10),
                    hoverinfo='text',
                    name=style['name'],
                    showlegend=True
                )
        
        # 创建图形
        fig = go.Figure(data=edge_trace + list(node_traces.values()))
        
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                xanchor='center',
                font=dict(size=20)
            ),
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            height=700
        )
        
        return fig
    
    def save_interactive_html(self, output_path: str, 
                            layout: str = 'spring',
                            title: str = '知识图谱'):
        """
        保存交互式HTML文件
        
        Args:
            output_path: 输出路径
            layout: 布局算法
            title: 图标题
        """
        fig = self.create_plotly_figure(layout, title)
        
        try:
            fig.write_html(output_path)
            logger.info(f"知识图谱已保存到: {output_path}")
        except Exception as e:
            logger.error(f"保存HTML失败: {e}")
            raise
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """
        获取图谱统计信息
        
        Returns:
            统计信息字典
        """
        node_types = {}
        for node, data in self.graph.nodes(data=True):
            node_type = data.get('type', 'unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        return {
            'total_nodes': len(self.graph.nodes),
            'total_edges': len(self.graph.edges),
            'node_types': node_types,
            'density': nx.density(self.graph),
            'is_connected': nx.is_connected(self.graph)
        }
    
    def export_graph_data(self, output_path: str):
        """
        导出图谱数据为JSON
        
        Args:
            output_path: 输出路径
        """
        data = {
            'nodes': [
                {
                    'id': node,
                    **self.graph.nodes[node]
                }
                for node in self.graph.nodes
            ],
            'edges': [
                {
                    'source': edge[0],
                    'target': edge[1],
                    **self.graph.edges[edge]
                }
                for edge in self.graph.edges
            ]
        }
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"图谱数据已导出到: {output_path}")
        except Exception as e:
            logger.error(f"导出图谱数据失败: {e}")
            raise


def create_visualizer(config: Dict[str, Any]) -> KnowledgeGraphVisualizer:
    """
    工厂函数:创建可视化器
    
    Args:
        config: 配置字典
        
    Returns:
        可视化器实例
    """
    return KnowledgeGraphVisualizer(config)


if __name__ == "__main__":
    # 测试代码
    import sys
    sys.path.append("..")
    from config import VISUALIZATION_CONFIG, QUESTION_DB
    from data_management.question_db import create_question_database
    
    logging.basicConfig(level=logging.INFO)
    
    # 加载题库
    db = create_question_database(str(QUESTION_DB))
    questions = db.get_all_questions()
    
    # 创建可视化器
    visualizer = create_visualizer(VISUALIZATION_CONFIG)
    
    # 构建图谱
    visualizer.build_graph_from_questions(questions)
    
    # 获取统计信息
    stats = visualizer.get_graph_statistics()
    print(f"图谱统计: {stats}")
    
    # 保存HTML
    visualizer.save_interactive_html('/home/claude/test_kg.html')
    print("可视化文件已保存")
