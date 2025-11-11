# -*- coding: utf-8 -*-
"""
Copyright (c) 2025 AI系统及应用课题组@SRIBD

基于LLM和知识图谱协同的个性化出题系统 (PersonalExam)
Personalized Question Generation System Based on LLM and Knowledge Graph Collaboration

"""

import gradio as gr
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class SmartEducationUI:
    """智能教育系统UI"""
    
    def __init__(self, system_core):
        self.system = system_core
        logger.info("✅ 智能教育UI初始化完成")
    
    def create_interface(self) -> gr.Blocks:
        """创建UI界面"""
        
        with gr.Blocks(title="个性化出题系统", theme=gr.themes.Soft()) as interface:
            
            gr.Markdown("""
            # 🧠 基于LLM和知识图谱协同的个性化出题系统
            """)
            
            with gr.Tabs():
                # Tab 1: 智能测评
                with gr.Tab("🎯 智能测评"):
                    self._create_smart_assessment_tab()
                
                # Tab 2: 学习分析
                with gr.Tab("📊 学习分析"):
                    self._create_analysis_tab()
                
                # Tab 3: 知识图谱
                with gr.Tab("🕸️ 知识图谱"):
                    self._create_knowledge_graph_tab()
                
                # Tab 4: 系统管理
                with gr.Tab("⚙️ 系统管理"):
                    self._create_management_tab()
        
        return interface
    
    def _create_smart_assessment_tab(self):
        """创建智能测评标签页"""
        
        gr.Markdown("### 🚀 开始测评")
        gr.Markdown("""

        """)
        
        # 基本设置
        with gr.Row():
            student_id_input = gr.Textbox(
                label="🆔 学生ID",
                placeholder="请输入学生ID（如 student_001）",
                value="student_001"
            )
            num_questions = gr.Slider(
                minimum=5,
                maximum=20,
                value=10,
                step=1,
                label="📝 题目数量"
            )
        
        start_btn = gr.Button("🚀 开始智能测评", variant="primary", size="lg")
        
        # 学生档案预览（美化）
        with gr.Accordion("📋 我的学习档案", open=False):
            profile_display = gr.Markdown("暂无数据，开始测评后将显示您的学习档案")
        
        gr.Markdown("---")
        
        # 测评区域
        session_state = gr.State(value=None)
        
        with gr.Column(visible=False) as quiz_area:
            # 进度和AI状态显示
            with gr.Row():
                progress_text = gr.Markdown("### 📊 进度: 1/10")
                ai_status = gr.Markdown("**🤖 AI状态:** 待命中")
            
            current_kp_text = gr.Markdown("**当前知识点:** 等待加载...")
            
            # 题目显示
            question_text = gr.Textbox(
                label="📝 题目",
                lines=6,
                interactive=False
            )
            
            # 答案输入
            answer_input = gr.Textbox(
                label="✍️ 你的答案",
                lines=3,
                placeholder="请输入你的答案..."
            )
            
            # 按钮
            with gr.Row():
                submit_answer_btn = gr.Button("✓ 提交答案", variant="primary")
                next_question_btn = gr.Button("→ 下一题", visible=False)
            
            # 反馈区域
            feedback_box = gr.Markdown("", visible=False)
        
        # 报告区域
        with gr.Column(visible=False) as report_area:
            gr.Markdown("### 📊 智能评估报告")
            gr.Markdown("*由盘古7B AI生成*")
            report_display = gr.Textbox(
                label="详细报告",
                lines=30,
                interactive=False
            )
            restart_btn = gr.Button("🔄 重新开始测评", variant="primary", size="lg")
        
        # 事件绑定
        start_btn.click(
            fn=self._start_smart_assessment,
            inputs=[student_id_input, num_questions],
            outputs=[
                session_state, quiz_area, report_area, question_text, 
                progress_text, current_kp_text, answer_input, profile_display,
                submit_answer_btn, next_question_btn, feedback_box, ai_status
            ]
        )
        
        submit_answer_btn.click(
            fn=self._submit_answer,
            inputs=[session_state, answer_input],
            outputs=[
                session_state, feedback_box, submit_answer_btn, 
                next_question_btn, answer_input, ai_status
            ]
        )
        
        next_question_btn.click(
            fn=self._next_question,
            inputs=[session_state],
            outputs=[
                session_state, question_text, progress_text, current_kp_text,
                feedback_box, submit_answer_btn, next_question_btn,
                answer_input, quiz_area, report_area, report_display, ai_status
            ]
        )
        
        restart_btn.click(
            fn=self._restart_assessment,
            outputs=[
                session_state, quiz_area, report_area, answer_input,
                submit_answer_btn, next_question_btn, feedback_box,
                progress_text, current_kp_text, question_text, ai_status
            ]
        )
    
    def _create_analysis_tab(self):
        """创建学习分析标签页"""
        
        gr.Markdown("### 📊 学习数据分析")
        gr.Markdown("*基于BKT算法的精准掌握度分析*")
        
        with gr.Row():
            student_id_for_analysis = gr.Textbox(
                label="学生ID",
                placeholder="输入学生ID查看分析",
                value="student_001"
            )
            analyze_btn = gr.Button("🔍 分析", variant="primary")
        
        # 整体概况
        with gr.Row():
            with gr.Column():
                overall_stats = gr.Markdown("### 📈 整体掌握度\n\n暂无数据")
            with gr.Column():
                weak_points_display = gr.Markdown("### ⚠️ 薄弱知识点\n\n暂无数据")
        
        # 详细档案
        gr.Markdown("### 📋 详细学习档案")
        detailed_profile = gr.Markdown("暂无数据")
        
        analyze_btn.click(
            fn=self._analyze_student,
            inputs=[student_id_for_analysis],
            outputs=[overall_stats, weak_points_display, detailed_profile]
        )
    
    def _create_knowledge_graph_tab(self):
        """创建知识图谱标签页"""
        
        gr.Markdown("### 🕸️ 知识图谱可视化")
        gr.Markdown("*展示题目、知识点和难度之间的关系网络*")
        
        with gr.Row():
            layout_choice = gr.Radio(
                choices=["spring", "circular", "kamada_kawai"],
                value="spring",
                label="布局算法",
                info="选择图谱的布局方式"
            )
            refresh_btn = gr.Button("🔄 刷新图谱", variant="primary")
        
        # 知识图谱展示（初始化时自动加载）
        initial_fig, initial_stats = self._refresh_knowledge_graph("spring")
        kg_plot = gr.Plot(label="知识图谱", value=initial_fig)
        
        # 图谱统计信息
        kg_stats = gr.Markdown(value=initial_stats)
        
        # 事件绑定
        refresh_btn.click(
            fn=self._refresh_knowledge_graph,
            inputs=[layout_choice],
            outputs=[kg_plot, kg_stats]
        )
        
        # 布局选择变化时自动刷新
        layout_choice.change(
            fn=self._refresh_knowledge_graph,
            inputs=[layout_choice],
            outputs=[kg_plot, kg_stats]
        )
    
    def _create_management_tab(self):
        """创建系统管理标签页"""
        
        gr.Markdown("### ⚙️ 系统管理")
        
        # 题库管理
        with gr.Tab("📚 题库管理"):
            gr.Markdown("#### 导入题目")
            
            json_file = gr.File(label="选择JSON文件", file_types=[".json"])
            import_btn = gr.Button("导入", variant="primary")
            import_status = gr.Textbox(label="导入状态", interactive=False)
            
            import_btn.click(
                fn=self._import_questions,
                inputs=[json_file],
                outputs=[import_status]
            )
            
            gr.Markdown("#### 题库统计")
            refresh_stats_btn = gr.Button("🔄 刷新统计")
            stats_display = gr.Markdown("暂无统计")
            
            refresh_stats_btn.click(
                fn=self._get_stats,
                outputs=[stats_display]
            )
        
        # 系统信息
        with gr.Tab("ℹ️ 系统信息"):
            system_info = gr.Textbox(
                label="系统状态",
                value=self.system.get_system_info(),
                lines=25,
                interactive=False
            )
            
            with gr.Row():
                reload_btn = gr.Button("🔄 重新加载模型")
                clear_cache_btn = gr.Button("🗑️ 清除缓存")
            
            operation_status = gr.Textbox(label="操作状态", interactive=False)
            
            reload_btn.click(
                fn=self._reload_models,
                outputs=[operation_status]
            )
            
            clear_cache_btn.click(
                fn=self._clear_cache,
                outputs=[operation_status]
            )
    
    # ==================== 回调函数 ====================
    
    def _start_smart_assessment(self, student_id: str, num: int):
        """开始智能测评"""
        try:
            logger.info(f"🚀 学生 {student_id} 开始测评")
            
            # 获取学生档案
            profile = self.system.bkt_algorithm.generate_student_profile(student_id)
            profile_md = self._format_profile_markdown(profile)
            
            # 开始测评
            session = self.system.start_smart_assessment(student_id, int(num))
            
            if session is None:
                return (
                    None, gr.update(visible=False), gr.update(visible=False),
                    "无法开始测评", "进度: 0/0", "知识点: N/A", "", 
                    profile_md, gr.update(), gr.update(), gr.update(visible=False),
                    "**🤖 AI状态:** 错误"
                )
            
            question = session['current_question']
            major = session['current_major_point']
            minor = session['current_minor_point']
            
            progress_md = f"### 📊 进度: {session['current_index']}/{session['total_questions']}"
            kp_md = f"**当前知识点:** {major} → {minor}"
            ai_status_md = "**🤖 AI状态:** 题目已选择（基于BKT算法）"
            
            return (
                session,
                gr.update(visible=True),   # quiz_area
                gr.update(visible=False),  # report_area
                question['问题'],
                progress_md,
                kp_md,
                "",                        # answer_input
                profile_md,
                gr.update(visible=True),   # submit_answer_btn
                gr.update(visible=False),  # next_question_btn
                gr.update(visible=False),  # feedback_box
                ai_status_md
            )
        except Exception as e:
            logger.error(f"开始测评失败: {e}")
            return (
                None, gr.update(visible=False), gr.update(visible=False),
                f"错误: {str(e)}", "进度: 0/0", "知识点: N/A", "", 
                "暂无数据", gr.update(), gr.update(), gr.update(visible=False),
                "**🤖 AI状态:** 错误"
            )
    
    def _submit_answer(self, session, answer):
        """提交答案"""
        if session is None:
            return (
                session, "请先开始测评", gr.update(), gr.update(), "",
                "**🤖 AI状态:** 待命中"
            )
        
        try:
            logger.info(f"📝 提交答案，正在使用盘古7B评估...")
            
            session = self.system.submit_answer(session, answer)
            last_result = session['last_result']
            
            # 构建反馈（美化）
            feedback = f"""
### 🎯 答题反馈

#### 📝 你的答案
{answer}

#### ✅ 标准答案
{last_result['question']['答案']}

#### 🤖 盘古7B判定
{'✅ **正确！**' if last_result['is_correct'] else '❌ **错误**'}

#### 💬 评判理由
{last_result['check_reason']}

#### 📚 知识点
{last_result['major_point']} → {last_result['minor_point']}

#### 📊 掌握度变化
- **答题前:** {last_result['mastery_before']:.1%}
- **答题后:** {last_result['mastery_after']:.1%}
- **变化:** {last_result['mastery_change']:+.1%}

#### 💡 解析
{last_result['question']['解析']}

---
*点击"下一题"继续测评*
"""
            
            ai_status = "**🤖 AI状态:** 盘古7B评估完成 ✓"
            
            return (
                session,
                gr.update(value=feedback, visible=True),
                gr.update(visible=False),   # 隐藏提交按钮
                gr.update(visible=True),    # 显示下一题按钮
                "",                         # 清空输入框
                ai_status
            )
        except Exception as e:
            logger.error(f"提交答案失败: {e}")
            return (
                session, f"❌ 错误: {str(e)}", 
                gr.update(), gr.update(), answer,
                "**🤖 AI状态:** 评估失败"
            )
    
    def _next_question(self, session):
        """下一题"""
        if session is None:
            return (
                None, "", "进度: 0/0", "知识点: N/A", 
                gr.update(visible=False), gr.update(visible=True), 
                gr.update(visible=False), "", 
                gr.update(visible=True), gr.update(visible=False), "",
                "**🤖 AI状态:** 待命中"
            )
        
        try:
            # 检查是否完成
            if session['current_index'] >= session['total_questions']:
                logger.info("📊 测评完成，正在生成报告...")
                
                # 使用盘古7B生成报告
                report = self.system.generate_report(session)
                
                return (
                    session,
                    "",  # question_text
                    f"### 📊 进度: {session['current_index']}/{session['total_questions']} (已完成)",
                    "**测评已完成**",
                    gr.update(visible=False),    # feedback_box
                    gr.update(visible=False),    # submit_answer_btn
                    gr.update(visible=False),    # next_question_btn
                    "",                          # answer_input
                    gr.update(visible=False),    # quiz_area
                    gr.update(visible=True),     # report_area
                    report,
                    "**🤖 AI状态:** 报告已生成（盘古7B）"
                )
            
            # 加载下一题
            session = self.system.next_question(session)
            question = session['current_question']
            major = session['current_major_point']
            minor = session['current_minor_point']
            
            progress_md = f"### 📊 进度: {session['current_index']}/{session['total_questions']}"
            kp_md = f"**当前知识点:** {major} → {minor}"
            ai_status = "**🤖 AI状态:** 已选择下一题（智能推荐）"
            
            return (
                session,
                question['问题'],
                progress_md,
                kp_md,
                gr.update(visible=False),    # feedback_box
                gr.update(visible=True),     # submit_answer_btn
                gr.update(visible=False),    # next_question_btn
                "",                          # answer_input
                gr.update(visible=True),     # quiz_area
                gr.update(visible=False),    # report_area
                "",                          # report_display
                ai_status
            )
        except Exception as e:
            logger.error(f"加载下一题失败: {e}")
            return (
                session, f"错误: {str(e)}", "进度: N/A", "知识点: N/A", 
                gr.update(visible=False), gr.update(visible=True), 
                gr.update(visible=False), "", 
                gr.update(visible=True), gr.update(visible=False), "",
                "**🤖 AI状态:** 错误"
            )
    
    def _restart_assessment(self):
        """重新开始测评（完整重置）"""
        logger.info("🔄 重置测评状态")
        return (
            None,                           # session_state
            gr.update(visible=False),       # quiz_area
            gr.update(visible=False),       # report_area
            "",                             # answer_input
            gr.update(visible=True),        # submit_answer_btn
            gr.update(visible=False),       # next_question_btn
            gr.update(visible=False),       # feedback_box
            "### 📊 进度: 0/0",            # progress_text
            "**当前知识点:** 请开始测评",  # current_kp_text
            "",                             # question_text (清空题目)
            "**🤖 AI状态:** 待命中"        # ai_status
        )
    
    def _analyze_student(self, student_id: str):
        """分析学生（美化版）"""
        try:
            profile = self.system.bkt_algorithm.generate_student_profile(student_id)
            
            # 整体统计 Markdown
            overall_md = f"""
### 📈 整体学习状况

| 指标 | 数值 |
|------|------|
| 📊 整体掌握度 | **{profile['overall_mastery']:.1%}** |
| 📚 已学知识点 | {profile['total_knowledge_points']} 个 |
| ✍️ 累计答题数 | {profile['total_answers']} 题 |
| 🚀 学习潜力 | {profile.get('learning_potential', '未知')} |

---
"""
            
            # 薄弱点 Markdown
            weak_points = profile['weak_points']
            if weak_points:
                weak_md = "### ⚠️ 薄弱知识点\n\n需要重点加强的知识点：\n\n"
                for i, (major, minor, mastery) in enumerate(weak_points[:5], 1):
                    bar = self._create_mastery_bar(mastery)
                    weak_md += f"{i}. **{major} / {minor}**\n   {bar} {mastery:.1%}\n\n"
            else:
                weak_md = "### ⚠️ 薄弱知识点\n\n✅ 无明显薄弱点，继续保持！"
            
            # 详细档案 Markdown
            detail_md = self._format_profile_markdown(profile)
            
            return overall_md, weak_md, detail_md
            
        except Exception as e:
            logger.error(f"分析失败: {e}")
            error_md = f"### ❌ 错误\n\n{str(e)}"
            return error_md, error_md, error_md
    
    def _format_profile_markdown(self, profile: Dict[str, Any]) -> str:
        """格式化学生档案为美化的Markdown"""
        md = f"""
## 👤 学生学习档案

### 📊 基本信息
| 项目 | 内容 |
|------|------|
| 🆔 学生ID | {profile.get('student_id', 'N/A')} |
| 📈 整体掌握度 | **{profile.get('overall_mastery', 0):.1%}** |
| 📚 已学知识点 | {profile.get('total_knowledge_points', 0)} 个 |
| ✍️ 累计答题数 | {profile.get('total_answers', 0)} 题 |

### 🎯 学习能力画像
"""
        
        # 学习潜力
        potential = profile.get('learning_potential', '未知')
        potential_icon = {
            '高': '🚀',
            '中等': '📈',
            '需要加强': '💪',
            '未知': '❓'
        }.get(potential, '📊')
        md += f"\n**{potential_icon} 学习潜力:** {potential}\n"
        
        # 学习特征
        if 'learning_characteristics' in profile:
            char = profile['learning_characteristics']
            md += f"\n**📖 难度偏好:** {char.get('difficulty_preference', '未知')}\n"
            
            stability = char.get('learning_stability', 0)
            stability_bar = self._create_mastery_bar(stability)
            md += f"**💎 学习稳定性:** {stability_bar} {stability:.1%}\n"
        
        # 优势知识点
        strengths = profile.get('strengths', [])
        if strengths:
            md += "\n### ✅ 优势知识点\n\n"
            for major, minor, mastery in strengths[:5]:
                bar = self._create_mastery_bar(mastery)
                md += f"- **{major} / {minor}**\n  {bar} {mastery:.1%}\n"
        
        # 薄弱知识点
        weak_points = profile.get('weak_points', [])
        if weak_points:
            md += "\n### ⚠️ 需要加强的知识点\n\n"
            for major, minor, mastery in weak_points[:5]:
                bar = self._create_mastery_bar(mastery)
                md += f"- **{major} / {minor}**\n  {bar} {mastery:.1%} ← 需要加强\n"
        
        # 知识点详情
        knowledge_points = profile.get('knowledge_points', {})
        if knowledge_points:
            md += "\n### 📚 知识点掌握详情\n\n"
            for major, minors in knowledge_points.items():
                md += f"\n#### 📖 {major}\n\n"
                for minor, details in minors.items():
                    mastery = details.get('mastery', 0)
                    total_ans = details.get('total_answers', 0)
                    recent_acc = details.get('recent_accuracy', 0)
                    
                    bar = self._create_mastery_bar(mastery)
                    md += f"**{minor}**\n"
                    md += f"- 掌握度: {bar} {mastery:.1%}\n"
                    md += f"- 答题数: {total_ans} 题\n"
                    md += f"- 近期准确率: {recent_acc:.1%}\n\n"
        
        return md
    
    def _create_mastery_bar(self, mastery: float, length: int = 20) -> str:
        """创建掌握度可视化条"""
        filled = int(mastery * length)
        empty = length - filled
        
        # 根据掌握度选择颜色（使用emoji）
        if mastery >= 0.7:
            bar = '🟩' * filled + '⬜' * empty
        elif mastery >= 0.4:
            bar = '🟨' * filled + '⬜' * empty
        else:
            bar = '🟥' * filled + '⬜' * empty
        
        return bar
    
    def _import_questions(self, file_obj):
        """导入题目"""
        if file_obj is None:
            return "请选择文件"
        
        try:
            count = self.system.import_questions(file_obj.name)
            return f"✅ 成功导入 {count} 道题目"
        except Exception as e:
            return f"❌ 导入失败: {str(e)}"
    
    def _get_stats(self):
        """获取统计"""
        try:
            stats = self.system.get_database_statistics()
            
            md = f"""
## 📊 题库统计信息

### 📈 基本数据
| 指标 | 数值 |
|------|------|
| 📚 总题目数 | **{stats['总题目数']}** 道 |
| 📖 知识点大类数 | {len(stats['知识点大类分布'])} 个 |
| 📝 知识点小类数 | {len(stats['知识点小类分布'])} 个 |

### 📊 知识点大类分布
"""
            for kp, count in sorted(stats['知识点大类分布'].items(), 
                                   key=lambda x: x[1], reverse=True):
                percentage = count / stats['总题目数'] * 100
                bar = '█' * int(percentage / 5)
                md += f"- **{kp}**: {count} 题 ({percentage:.1f}%) {bar}\n"
            
            md += "\n### 📊 难度分布\n"
            for diff, count in stats['难度分布'].items():
                percentage = count / stats['总题目数'] * 100
                bar = '█' * int(percentage / 5)
                md += f"- **{diff}**: {count} 题 ({percentage:.1f}%) {bar}\n"
            
            return md
        except Exception as e:
            return f"### ❌ 错误\n\n{str(e)}"
    
    def _reload_models(self):
        """重新加载模型"""
        try:
            self.system.reload_models()
            return "✅ 盘古7B模型重新加载成功"
        except Exception as e:
            return f"❌ 加载失败: {str(e)}"
    
    def _clear_cache(self):
        """清除缓存"""
        try:
            self.system.clear_cache()
            return "✅ NPU缓存已清除"
        except Exception as e:
            return f"❌ 清除失败: {str(e)}"
    
    def _refresh_knowledge_graph(self, layout: str):
        """刷新知识图谱"""
        try:
            # 获取图谱可视化
            fig = self.system.visualizer.create_plotly_figure(
                layout=layout,
                title="知识图谱 - 题目与知识点关系网络"
            )
            
            # 获取图谱统计
            stats = self.system.visualizer.get_graph_statistics()
            
            # 格式化统计信息为 Markdown
            stats_md = f"""
### 📊 图谱统计信息

| 指标 | 数值 |
|------|------|
| 📊 总节点数 | **{stats['total_nodes']}** 个 |
| 🔗 总边数 | **{stats['total_edges']}** 条 |
| 📈 图谱密度 | {stats['density']:.4f} |
| 🔄 连通性 | {'✅ 连通' if stats['is_connected'] else '❌ 非连通'} |

### 📋 节点类型分布
"""
            for node_type, count in stats['node_types'].items():
                type_name = {'knowledge': '知识点', 'difficulty': '难度', 'question': '题目'}.get(node_type, node_type)
                stats_md += f"- **{type_name}**: {count} 个\n"
            
            return fig, stats_md
            
        except Exception as e:
            logger.error(f"刷新知识图谱失败: {e}")
            error_md = f"### ❌ 错误\n\n加载知识图谱失败: {str(e)}"
            return None, error_md


def create_ui(system_core) -> gr.Blocks:
    """创建UI界面"""
    ui = SmartEducationUI(system_core)
    return ui.create_interface()


if __name__ == "__main__":
    print("请从主程序运行UI")