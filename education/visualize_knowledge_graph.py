#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
知识图谱可视化脚本
支持多种可视化方式：Plotly 3D、Plotly 2D、NetworkX静态图
"""

import sys
import logging
from pathlib import Path
import pickle
import argparse

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import networkx as nx
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class KnowledgeGraphVisualizer:
    """知识图谱可视化器"""
    
    def __init__(self, kg_path: str):
        """
        Args:
            kg_path: 知识图谱缓存文件路径
        """
        self.kg_path = Path(kg_path)
        self.graph = None
        
        # 节点颜色配置
        self.node_colors = {
            'question': '#95E1D3',      # 青绿色 - 题目
            'major_point': '#FF6B6B',   # 红色 - 知识点大类
            'minor_point': '#4ECDC4',   # 青色 - 知识点小类
            'concept': '#FFD93D',       # 黄色 - 概念
            'method': '#A8E6CF',        # 浅绿色 - 方法
            'default': '#CCCCCC'        # 灰色 - 其他
        }
        
        self.node_sizes = {
            'question': 8,
            'major_point': 25,
            'minor_point': 18,
            'concept': 15,
            'method': 12,
            'default': 10
        }

        self.edge_colors = {
            'tests': '#888888',         # 灰色 - 测试关系
            'belongs_to': '#FF6B6B',    # 红色 - 归属关系
            'prerequisite': '#4ECDC4',  # 青色 - 前置关系
            'leads_to': '#FFD93D',      # 黄色 - 后续关系
            'involves': '#A8E6CF',      # 浅绿色 - 涉及关系
            'uses': '#B8A1E5',          # 紫色 - 使用关系
            'default': '#DDDDDD'
        }
        
    def load_graph(self):
        try:
            logger.info(f"加载知识图谱: {self.kg_path}")
            
            if not self.kg_path.exists():
                raise FileNotFoundError(f"知识图谱文件不存在: {self.kg_path}")
            
            with open(self.kg_path, 'rb') as f:
                self.graph = pickle.load(f)
            
            logger.info(f"✅ 知识图谱加载成功:")
            logger.info(f"   - 节点数: {self.graph.number_of_nodes()}")
            logger.info(f"   - 边数: {self.graph.number_of_edges()}")
            
            # 统计节点类型
            node_types = {}
            for node, data in self.graph.nodes(data=True):
                node_type = data.get('type', 'unknown')
                node_types[node_type] = node_types.get(node_type, 0) + 1
            
            logger.info(f"   - 节点类型分布:")
            for node_type, count in sorted(node_types.items()):
                logger.info(f"     • {node_type}: {count}")
            
            return True
            
        except Exception as e:
            logger.error(f"加载知识图谱失败: {e}")
            return False
    
    def get_statistics(self) -> dict:
        if not self.graph:
            return {}
        
        stats = {
            'nodes': self.graph.number_of_nodes(),
            'edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'is_connected': nx.is_weakly_connected(self.graph),
        }
        
        node_types = {}
        for node, data in self.graph.nodes(data=True):
            node_type = data.get('type', 'unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1
        stats['node_types'] = node_types
        
        edge_types = {}
        for u, v, data in self.graph.edges(data=True):
            edge_type = data.get('relation', 'unknown')
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
        stats['edge_types'] = edge_types
        
        return stats
    
    def visualize_plotly_3d(self, output_path: str = "knowledge_graph_3d.html",
                           show_edges: bool = True, 
                           max_nodes: int = None):

        logger.info("🎨 正在生成 Plotly 3D 可视化...")
        
        graph = self.graph
        if max_nodes and self.graph.number_of_nodes() > max_nodes:
            logger.info(f"⚠️  节点数 {self.graph.number_of_nodes()} 超过限制 {max_nodes}，进行采样...")

            important_nodes = [n for n, d in self.graph.nodes(data=True) 
                             if d.get('type') in ['major_point', 'minor_point', 'concept', 'method']]
            question_nodes = [n for n, d in self.graph.nodes(data=True) 
                            if d.get('type') == 'question']

            import random
            remaining = max_nodes - len(important_nodes)
            if remaining > 0:
                sampled_questions = random.sample(question_nodes, min(remaining, len(question_nodes)))
                nodes_to_keep = important_nodes + sampled_questions
            else:
                nodes_to_keep = important_nodes[:max_nodes]
            
            graph = self.graph.subgraph(nodes_to_keep).copy()
            logger.info(f"✅ 采样后节点数: {graph.number_of_nodes()}")
        
        logger.info("📐 计算节点位置（3D spring layout）...")
        pos = nx.spring_layout(graph, dim=3, k=0.5, iterations=50)
        
        edge_traces = []
        node_traces = []

        if show_edges:
            logger.info("🔗 创建边轨迹...")
            edge_x, edge_y, edge_z = [], [], []
            
            for u, v in graph.edges():
                x0, y0, z0 = pos[u]
                x1, y1, z1 = pos[v]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                edge_z.extend([z0, z1, None])
            
            edge_trace = go.Scatter3d(
                x=edge_x, y=edge_y, z=edge_z,
                mode='lines',
                line=dict(color='#888', width=1),
                hoverinfo='none',
                showlegend=False
            )
            edge_traces.append(edge_trace)

        logger.info("创建节点轨迹...")
        node_groups = {}
        for node, data in graph.nodes(data=True):
            node_type = data.get('type', 'default')
            if node_type not in node_groups:
                node_groups[node_type] = {'nodes': [], 'colors': [], 'sizes': [], 'texts': []}
            
            x, y, z = pos[node]
            node_groups[node_type]['nodes'].append((x, y, z))

            name = data.get('name', node)
            if len(str(name)) > 30:
                name = str(name)[:27] + "..."
            node_groups[node_type]['texts'].append(name)

        type_names = {
            'question': '题目',
            'major_point': '知识点大类',
            'minor_point': '知识点小类',
            'concept': '概念',
            'method': '方法'
        }
        
        for node_type, group_data in node_groups.items():
            if not group_data['nodes']:
                continue
            
            x_vals, y_vals, z_vals = zip(*group_data['nodes'])
            
            node_trace = go.Scatter3d(
                x=x_vals, y=y_vals, z=z_vals,
                mode='markers+text',
                marker=dict(
                    size=self.node_sizes.get(node_type, 10),
                    color=self.node_colors.get(node_type, '#CCCCCC'),
                    line=dict(color='white', width=0.5)
                ),
                text=group_data['texts'],
                textposition='top center',
                textfont=dict(size=8),
                hoverinfo='text',
                hovertext=group_data['texts'],
                name=type_names.get(node_type, node_type),
                showlegend=True
            )
            node_traces.append(node_trace)

        fig = go.Figure(data=edge_traces + node_traces)

        stats = self.get_statistics()
        title_text = f"知识图谱 3D 可视化<br><sub>节点: {graph.number_of_nodes()} | 边: {graph.number_of_edges()}</sub>"
        
        fig.update_layout(
            title=dict(text=title_text, x=0.5, xanchor='center'),
            showlegend=True,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=40),
            scene=dict(
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                zaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                bgcolor='white'
            ),
            height=800
        )

        fig.write_html(output_path)
        logger.info(f"✅ 3D 可视化已保存到: {output_path}")
        
        return fig
    
    def visualize_plotly_2d(self, output_path: str = "knowledge_graph_2d.html",
                           layout: str = 'spring',
                           max_nodes: int = None):

        logger.info(f"🎨 正在生成 Plotly 2D 可视化（布局: {layout}）...")
        
        graph = self.graph
        if max_nodes and self.graph.number_of_nodes() > max_nodes:
            logger.info(f"⚠️  节点数过多，采样到 {max_nodes} 个节点...")
            import random
            important_nodes = [n for n, d in self.graph.nodes(data=True) 
                             if d.get('type') in ['major_point', 'minor_point', 'concept', 'method']]
            question_nodes = [n for n, d in self.graph.nodes(data=True) 
                            if d.get('type') == 'question']
            
            remaining = max_nodes - len(important_nodes)
            if remaining > 0:
                sampled = important_nodes + random.sample(question_nodes, min(remaining, len(question_nodes)))
            else:
                sampled = important_nodes[:max_nodes]
            
            graph = self.graph.subgraph(sampled).copy()

        logger.info(f"📐 计算节点位置（{layout} layout）...")
        if layout == 'spring':
            pos = nx.spring_layout(graph, k=2, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(graph)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(graph)
        else:
            pos = nx.spring_layout(graph)
        
        edge_traces = []
        for u, v, data in graph.edges(data=True):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            
            relation = data.get('relation', 'default')
            color = self.edge_colors.get(relation, '#DDDDDD')
            
            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=1, color=color),
                hoverinfo='none',
                showlegend=False
            )
            edge_traces.append(edge_trace)

        node_traces = []
        node_groups = {}
        
        for node, data in graph.nodes(data=True):
            node_type = data.get('type', 'default')
            if node_type not in node_groups:
                node_groups[node_type] = {'x': [], 'y': [], 'texts': []}
            
            x, y = pos[node]
            node_groups[node_type]['x'].append(x)
            node_groups[node_type]['y'].append(y)
            
            name = data.get('name', node)
            if len(str(name)) > 30:
                name = str(name)[:27] + "..."
            node_groups[node_type]['texts'].append(name)
        
        type_names = {
            'question': '题目',
            'major_point': '知识点大类',
            'minor_point': '知识点小类',
            'concept': '概念',
            'method': '方法'
        }
        
        for node_type, group_data in node_groups.items():
            if not group_data['x']:
                continue
            
            node_trace = go.Scatter(
                x=group_data['x'],
                y=group_data['y'],
                mode='markers+text',
                marker=dict(
                    size=self.node_sizes.get(node_type, 10),
                    color=self.node_colors.get(node_type, '#CCCCCC'),
                    line=dict(color='white', width=1)
                ),
                text=group_data['texts'],
                textposition='top center',
                textfont=dict(size=10),
                hoverinfo='text',
                hovertext=group_data['texts'],
                name=type_names.get(node_type, node_type),
                showlegend=True
            )
            node_traces.append(node_trace)
        
        # 创建图形
        fig = go.Figure(data=edge_traces + node_traces)
        
        title_text = f"知识图谱 2D 可视化<br><sub>节点: {graph.number_of_nodes()} | 边: {graph.number_of_edges()}</sub>"
        
        fig.update_layout(
            title=dict(text=title_text, x=0.5, xanchor='center'),
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            height=800
        )
        
        fig.write_html(output_path)
        logger.info(f"✅ 2D 可视化已保存到: {output_path}")
        
        return fig
    
    def visualize_matplotlib(self, output_path: str = "knowledge_graph_static.png",
                            layout: str = 'spring',
                            max_nodes: int = 500,
                            figsize: tuple = (20, 15)):

        logger.info("🎨 正在生成 Matplotlib 静态可视化...")
        
        # 设置中文字体
        try:
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
        except:
            logger.warning("⚠️  无法设置中文字体，可能显示乱码")
        
        # 节点采样
        graph = self.graph
        if self.graph.number_of_nodes() > max_nodes:
            logger.info(f"⚠️  节点数过多，采样到 {max_nodes} 个节点...")
            import random
            important_nodes = [n for n, d in self.graph.nodes(data=True) 
                             if d.get('type') in ['major_point', 'minor_point', 'concept', 'method']]
            question_nodes = [n for n, d in self.graph.nodes(data=True) 
                            if d.get('type') == 'question']
            
            remaining = max_nodes - len(important_nodes)
            if remaining > 0:
                sampled = important_nodes + random.sample(question_nodes, min(remaining, len(question_nodes)))
            else:
                sampled = important_nodes[:max_nodes]
            
            graph = self.graph.subgraph(sampled).copy()
        
        # 计算布局
        if layout == 'spring':
            pos = nx.spring_layout(graph, k=1.5, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(graph)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(graph)
        else:
            pos = nx.spring_layout(graph)
        
        # 创建画布
        fig, ax = plt.subplots(figsize=figsize, facecolor='white')
        ax.set_facecolor('white')
        ax.axis('off')
        
        # 绘制边
        nx.draw_networkx_edges(
            graph, pos,
            edge_color='#CCCCCC',
            width=0.5,
            alpha=0.5,
            ax=ax
        )
        
        # 按类型绘制节点
        node_types = set(data.get('type', 'default') for _, data in graph.nodes(data=True))
        
        type_names = {
            'question': '题目',
            'major_point': '知识点大类',
            'minor_point': '知识点小类',
            'concept': '概念',
            'method': '方法'
        }
        
        for node_type in node_types:
            nodes = [n for n, d in graph.nodes(data=True) if d.get('type', 'default') == node_type]
            if not nodes:
                continue
            
            nx.draw_networkx_nodes(
                graph, pos,
                nodelist=nodes,
                node_color=self.node_colors.get(node_type, '#CCCCCC'),
                node_size=self.node_sizes.get(node_type, 10) * 20,
                label=type_names.get(node_type, node_type),
                ax=ax
            )
        
        labels = {}
        for node, data in graph.nodes(data=True):
            if data.get('type') in ['major_point', 'minor_point', 'concept', 'method']:
                name = data.get('name', node)
                if len(str(name)) > 20:
                    name = str(name)[:17] + "..."
                labels[node] = name
        
        nx.draw_networkx_labels(
            graph, pos,
            labels,
            font_size=8,
            font_color='black',
            ax=ax
        )
        
        # 添加标题和图例
        plt.title(f"知识图谱可视化\n节点: {graph.number_of_nodes()} | 边: {graph.number_of_edges()}", 
                 fontsize=16, pad=20)
        plt.legend(loc='upper left', fontsize=10)
        
        # 保存
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"✅ 静态可视化已保存到: {output_path}")
        
        plt.close()
    
    def print_statistics(self):
        """打印统计信息"""
        stats = self.get_statistics()
        
        print("\n" + "=" * 70)
        print("📊 知识图谱统计信息")
        print("=" * 70)
        print(f"总节点数: {stats['nodes']}")
        print(f"总边数: {stats['edges']}")
        print(f"图密度: {stats['density']:.4f}")
        print(f"连通性: {'连通' if stats['is_connected'] else '非连通'}")
        
        print(f"\n节点类型分布:")
        for node_type, count in sorted(stats['node_types'].items()):
            percentage = count / stats['nodes'] * 100
            print(f"  {node_type:15s}: {count:4d} ({percentage:5.1f}%)")
        
        print(f"\n边类型分布:")
        for edge_type, count in sorted(stats['edge_types'].items()):
            percentage = count / stats['edges'] * 100 if stats['edges'] > 0 else 0
            print(f"  {edge_type:15s}: {count:4d} ({percentage:5.1f}%)")
        
        print("=" * 70 + "\n")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='知识图谱可视化工具')
    parser.add_argument('--kg-path', type=str, default='/data/weitianyu/teach_system/education/data/knowledge_graph.pkl',
                       help='知识图谱缓存文件路径')
    parser.add_argument('--mode', type=str, default='all',
                       choices=['2d', '3d', 'static', 'all'],
                       help='可视化模式')
    parser.add_argument('--layout', type=str, default='spring',
                       choices=['spring', 'circular', 'kamada_kawai'],
                       help='布局算法')
    parser.add_argument('--max-nodes', type=int, default=None,
                       help='最大显示节点数（None表示全部）')
    parser.add_argument('--output-dir', type=str, default='/data/weitianyu/teach_system/education/visualizations',
                       help='输出目录')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🎨 知识图谱可视化工具")
    print("=" * 70)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    visualizer = KnowledgeGraphVisualizer(args.kg_path)

    if not visualizer.load_graph():
        logger.error("❌ 无法加载知识图谱，退出")
        return

    visualizer.print_statistics()

    try:
        if args.mode in ['2d', 'all']:
            logger.info("🎯 生成 2D Plotly 可视化...")
            output_2d = output_dir / "knowledge_graph_2d.html"
            visualizer.visualize_plotly_2d(
                output_path=str(output_2d),
                layout=args.layout,
                max_nodes=args.max_nodes
            )
        
        if args.mode in ['3d', 'all']:
            logger.info("🎯 生成 3D Plotly 可视化...")
            output_3d = output_dir / "knowledge_graph_3d.html"
            visualizer.visualize_plotly_3d(
                output_path=str(output_3d),
                show_edges=True,
                max_nodes=args.max_nodes
            )
        
        if args.mode in ['static', 'all']:
            logger.info("🎯 生成 Matplotlib 静态可视化...")
            output_static = output_dir / "knowledge_graph_static.png"
            visualizer.visualize_matplotlib(
                output_path=str(output_static),
                layout=args.layout,
                max_nodes=500  # 静态图限制节点数
            )
        
        print("\n" + "=" * 70)
        print("✅ 可视化完成！")
        print("=" * 70)
        print(f"📁 输出目录: {output_dir.absolute()}")
        print(f"   - 2D 交互式: knowledge_graph_2d.html")
        print(f"   - 3D 交互式: knowledge_graph_3d.html")
        print(f"   - 静态图片: knowledge_graph_static.png")
        print("=" * 70)
        
    except Exception as e:
        logger.error(f"❌ 可视化失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()