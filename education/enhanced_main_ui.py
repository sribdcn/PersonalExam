# -*- coding: utf-8 -*-


import gradio as gr
import logging
from typing import Dict, Any, Optional
import plotly.graph_objects as go
import pandas as pd

logger = logging.getLogger(__name__)


# 自定义CSS样式
CUSTOM_CSS = """
/* 全局字体设置：中文新宋体，英文Times New Roman */
* {
    font-family: 'Times New Roman', 'NSimSun', '新宋体', serif !important;
}

/* 英文字符斜体，中文正常 */
*:lang(en), 
*[lang="en"],
*:not(:lang(zh)):not(:lang(zh-CN)):not(:lang(zh-TW)) {
    font-style: italic;
}

/* 确保中文字符不斜体 */
*:lang(zh),
*:lang(zh-CN),
*:lang(zh-TW),
*[lang="zh"],
*[lang="zh-CN"],
*[lang="zh-TW"] {
    font-style: normal !important;
}

/* Markdown内容字体 */
.markdown-text, .prose, .gr-markdown {
    font-family: 'Times New Roman', 'NSimSun', '新宋体', serif !important;
}

/* 输入框字体 */
input, textarea, .gr-input, .gr-textbox {
    font-family: 'Times New Roman', 'NSimSun', '新宋体', serif !important;
}

/* 按钮字体 */
button, .gr-button {
    font-family: 'Times New Roman', 'NSimSun', '新宋体', serif !important;
}

/* 标签字体 */
label, .gr-label {
    font-family: 'Times New Roman', 'NSimSun', '新宋体', serif !important;
}

/* 表格字体 */
table, .gr-dataframe {
    font-family: 'Times New Roman', 'NSimSun', '新宋体', serif !important;
}

/* 下拉框字体 */
select, .gr-dropdown {
    font-family: 'Times New Roman', 'NSimSun', '新宋体', serif !important;
}

/* Tab标签字体 */
.tabs, .tab-nav {
    font-family: 'Times New Roman', 'NSimSun', '新宋体', serif !important;
}

/* body基础字体 */
body {
    font-family: 'Times New Roman', 'NSimSun', '新宋体', serif !important;
}

/* Plotly图表标题和标签 */
.plotly .gtitle, .plotly text {
    font-family: 'Times New Roman', 'NSimSun', '新宋体', serif !important;
}

/* 针对特定英文元素设置斜体 */
code, pre, .code {
    font-style: italic;
}
"""


class EnhancedEducationUI:
    """增强版教育系统UI(含知识图谱可视化)"""
    
    def __init__(self, system_core, db_manager):
        self.system = system_core
        self.db = db_manager
        self.current_user = None
        logger.info("✅ 增强版UI初始化完成(含知识图谱)")
    
    def create_interface(self) -> gr.Blocks:
        """创建UI界面"""
        
        # 使用自定义CSS
        with gr.Blocks(
            title="智能教育系统", 
            theme=gr.themes.Soft(),
            css=CUSTOM_CSS  # 添加自定义CSS
        ) as interface:
            
            # 全局状态
            user_state = gr.State(value=None)
            
            gr.Markdown("""
            # 🧠 智能教育系统
            ## LLM和知识图谱的个性化学习平台
            """)
            
            # 登录/注册界面
            with gr.Column(visible=True) as login_register_area:
                gr.Markdown("## 🔐 用户登录")
                
                with gr.Row():
                    username_input = gr.Textbox(
                        label="用户名", 
                        placeholder="请输入用户名"
                    )
                    password_input = gr.Textbox(
                        label="密码", 
                        type="password", 
                        placeholder="请输入密码"
                    )
                
                with gr.Row():
                    login_btn = gr.Button("🔓 登录", variant="primary", size="lg")
                    register_btn = gr.Button("📝 注册新用户", variant="secondary", size="lg")
                
                login_status = gr.Markdown("")
                
                # 注册表单(默认隐藏)
                with gr.Column(visible=False) as register_form:
                    gr.Markdown("### 📝 用户注册")
                    
                    with gr.Row():
                        reg_username = gr.Textbox(label="用户名", placeholder="请输入用户名(6-20个字符)")
                        reg_password = gr.Textbox(label="密码", type="password", placeholder="请输入密码(至少6个字符)")
                    
                    with gr.Row():
                        reg_password_confirm = gr.Textbox(label="确认密码", type="password", placeholder="再次输入密码")
                        reg_realname = gr.Textbox(label="真实姓名", placeholder="请输入真实姓名(可选)")
                    
                    reg_role = gr.Radio(
                        choices=["student", "teacher"],
                        value="student",
                        label="账户类型",
                        info="选择学生或教师账户"
                    )
                    
                    with gr.Row():
                        confirm_register_btn = gr.Button("✅ 确认注册", variant="primary")
                        cancel_register_btn = gr.Button("❌ 取消", variant="secondary")
                    
                    register_status = gr.Markdown("")
            
            # 主界面(登录后显示)
            with gr.Column(visible=False) as main_area:
                # 用户信息栏
                with gr.Row():
                    user_info_display = gr.Markdown("")
                    logout_btn = gr.Button("🚪 退出登录", size="sm")
                
                # 学生界面
                with gr.Column(visible=False) as student_interface:
                    with gr.Tabs():
                        # 智能测评
                        with gr.Tab("🎯 智能测评"):
                            self._create_assessment_tab()
                        
                        # 我的学习
                        with gr.Tab("📊 我的学习"):
                            self._create_student_analysis_tab()
                        
                        # 知识图谱(新增)
                        with gr.Tab("🕸️ 知识图谱"):
                            self._create_knowledge_graph_tab_for_student()
                
                # 教师界面
                with gr.Column(visible=False) as teacher_interface:
                    with gr.Tabs():
                        # 题库管理
                        with gr.Tab("📚 题库管理"):
                            self._create_question_management_tab()
                        
                        # 学生管理
                        with gr.Tab("👥 学生管理"):
                            self._create_student_management_tab()
                        
                        # 知识图谱(新增)
                        with gr.Tab("🕸️ 知识图谱"):
                            self._create_knowledge_graph_tab_for_teacher()
                        
                        # 系统管理
                        with gr.Tab("⚙️ 系统管理"):
                            self._create_system_management_tab()
            
            # 事件绑定 - 登录
            login_btn.click(
                fn=self._handle_login,
                inputs=[username_input, password_input],
                outputs=[
                    user_state, login_register_area, main_area,
                    student_interface, teacher_interface,
                    user_info_display, login_status
                ]
            )
            
            # 事件绑定 - 显示注册表单
            register_btn.click(
                fn=lambda: (gr.update(visible=True), ""),
                outputs=[register_form, register_status]
            )
            
            # 事件绑定 - 确认注册
            confirm_register_btn.click(
                fn=self._handle_register,
                inputs=[reg_username, reg_password, reg_password_confirm, reg_realname, reg_role],
                outputs=[register_status, register_form, login_status]
            )
            
            # 事件绑定 - 取消注册
            cancel_register_btn.click(
                fn=lambda: (gr.update(visible=False), ""),
                outputs=[register_form, register_status]
            )
            
            # 事件绑定 - 退出登录
            logout_btn.click(
                fn=self._handle_logout,
                outputs=[
                    user_state, login_register_area, main_area,
                    student_interface, teacher_interface,
                    username_input, password_input, login_status, register_form
                ]
            )
        
        return interface
    
    
    def _handle_register(self, username: str, password: str, password_confirm: str, 
                        realname: str, role: str):
        """处理用户注册"""
        # 验证输入
        if not username or not password:
            return "❌ 用户名和密码不能为空!", gr.update(), ""
        
        if len(username) < 6 or len(username) > 20:
            return "❌ 用户名长度应在6-20个字符之间!", gr.update(), ""
        
        if len(password) < 6:
            return "❌ 密码长度至少6个字符!", gr.update(), ""
        
        if password != password_confirm:
            return "❌ 两次输入的密码不一致!", gr.update(), ""
        
        # 尝试创建用户
        success = self.db.create_user(username, password, role, realname if realname else None)
        
        if success:
            role_name = "学生" if role == "student" else "教师"
            return (
                f"✅ 注册成功!\n\n用户名: {username}\n类型: {role_name}\n\n请返回登录。",
                gr.update(visible=False),
                f"✅ 注册成功!请使用 {username} 登录。"
            )
        else:
            return "❌ 注册失败!用户名可能已存在,请更换用户名。", gr.update(), ""
    
    def _handle_login(self, username: str, password: str):
        """处理登录"""
        if not username or not password:
            return (
                None,
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                "",
                "❌ 请输入用户名和密码!"
            )
        
        user = self.db.verify_user(username, password)
        
        if user:
            self.current_user = user
            user_info_md = f"**👤 当前用户:** {user['real_name'] or user['username']} " \
                          f"({'👨‍🎓 学生' if user['role'] == 'student' else '👨‍🏫 教师'})"
            
            if user['role'] == 'student':
                return (
                    user,
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=True),
                    gr.update(visible=False),
                    user_info_md,
                    ""
                )
            else:  # teacher
                return (
                    user,
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=True),
                    user_info_md,
                    ""
                )
        else:
            return (
                None,
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                "",
                "❌ 用户名或密码错误!请检查后重试。"
            )
    
    def _handle_logout(self):
        """处理登出"""
        self.current_user = None
        return (
            None,
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            "",
            "",
            "✅ 已成功退出登录",
            gr.update(visible=False)
        )
    
    def _create_assessment_tab(self):
        """创建测评标签页(学生)"""
        gr.Markdown("### 🚀 开始智能测评")
    
        with gr.Row():
            num_questions = gr.Slider(
                minimum=5,
                maximum=20,
                value=10,
                step=1,
                label="📝 题目数量"
            )
    
        start_btn = gr.Button("🎯 开始测评", variant="primary", size="lg")
    
        # 测评区域
        session_state = gr.State(value=None)
    
        with gr.Column(visible=False) as quiz_area:
            progress_text = gr.Markdown("### 进度: 1/10")
            question_text = gr.Textbox(label="📄 题目", lines=8, interactive=False)
            answer_input = gr.Textbox(label="✏️ 你的答案", lines=3)
        
            with gr.Row():
                submit_btn = gr.Button("✓ 提交答案", variant="primary")
                next_btn = gr.Button("→ 下一题", visible=False)
        
            # 反馈框(初始隐藏)
            feedback_box = gr.Markdown("", visible=False)
    
        with gr.Column(visible=False) as report_area:
            gr.Markdown("### 📊 测评报告")
            report_display = gr.Textbox(label="详细报告", lines=30, interactive=False)
            restart_btn = gr.Button("🔄 重新测评", variant="primary")
    
        # 事件绑定
        start_btn.click(
            fn=self._start_assessment_for_current_user,
            inputs=[num_questions],
            outputs=[session_state, quiz_area, question_text, progress_text]
        )
    
        submit_btn.click(
            fn=self._submit_answer,
            inputs=[session_state, answer_input],
            outputs=[session_state, feedback_box, submit_btn, next_btn, answer_input]
        )
    
        next_btn.click(
            fn=self._next_question_fixed,  # 使用新的修复方法
            inputs=[session_state],
            outputs=[
                session_state, question_text, progress_text,
                feedback_box, submit_btn, next_btn, answer_input,
                quiz_area, report_area, report_display  # 🔧 添加这两个
            ]
        )
    
        # 🔧 修复：移除 quiz_area 和 report_area
        restart_btn.click(
            fn=self._restart_assessment,
            outputs=[
                session_state, answer_input,
                submit_btn, next_btn, feedback_box,
                progress_text, question_text, report_display
            ]
        )
    
    def _create_student_analysis_tab(self):
        """创建学生学习分析标签页"""
        gr.Markdown("### 📊 我的学习数据")
        
        refresh_btn = gr.Button("🔄 刷新数据", variant="primary")
        
        with gr.Row():
            overall_stats = gr.Markdown("### 📈 整体掌握度\n\n暂无数据")
            weak_points = gr.Markdown("### ⚠️ 薄弱知识点\n\n暂无数据")
        
        with gr.Row():
            radar_plot = gr.Plot(label="📊 掌握度雷达图")
        
        gr.Markdown("### 📝 最近答题历史")
        history_table = gr.Dataframe(
            headers=["题号", "知识点大类", "知识点小类", "是否正确", "掌握度变化", "答题时间"],
            interactive=False
        )
        
        refresh_btn.click(
            fn=self._load_student_data,
            outputs=[overall_stats, weak_points, radar_plot, history_table]
        )
    
    def _create_knowledge_graph_tab_for_student(self):
        """创建知识图谱标签页(学生版)"""
        gr.Markdown("### 🕸️ 知识图谱可视化")
        gr.Markdown("*探索题目、知识点之间的关联关系*")
        
        with gr.Row():
            kg_layout = gr.Radio(
                choices=["spring", "circular", "kamada_kawai"],
                value="spring",
                label="📐 布局算法",
                info="选择图谱的展示方式"
            )
            kg_dimension = gr.Radio(
                choices=["2D", "3D"],
                value="2D",
                label="📊 维度",
                info="2D更清晰,3D更立体"
            )
        
        with gr.Row():
            max_nodes_slider = gr.Slider(
                minimum=50,
                maximum=500,
                value=200,
                step=50,
                label="🔢 最大显示节点数",
                info="节点过多会影响性能"
            )
            show_edges_checkbox = gr.Checkbox(
                label="显示连线",
                value=True,
                info="隐藏连线可提升性能"
            )
        
        refresh_kg_btn = gr.Button("🔄 刷新图谱", variant="primary", size="lg")
        
        # 知识图谱显示区域
        kg_plot = gr.Plot(label="知识图谱", value=None)
        
        # 图谱统计信息
        kg_stats_display = gr.Markdown("### 📊 图谱统计\n\n点击刷新按钮加载图谱")
        
        # 绑定刷新事件
        refresh_kg_btn.click(
            fn=self._refresh_knowledge_graph,
            inputs=[kg_layout, kg_dimension, max_nodes_slider, show_edges_checkbox],
            outputs=[kg_plot, kg_stats_display]
        )
        
        # 布局和维度变化时自动刷新
        kg_layout.change(
            fn=self._refresh_knowledge_graph,
            inputs=[kg_layout, kg_dimension, max_nodes_slider, show_edges_checkbox],
            outputs=[kg_plot, kg_stats_display]
        )
        
        kg_dimension.change(
            fn=self._refresh_knowledge_graph,
            inputs=[kg_layout, kg_dimension, max_nodes_slider, show_edges_checkbox],
            outputs=[kg_plot, kg_stats_display]
        )
    
    def _create_knowledge_graph_tab_for_teacher(self):
        """创建知识图谱标签页(教师版 - 功能更丰富)"""
        gr.Markdown("### 🕸️ 知识图谱管理")
        gr.Markdown("*查看和管理题库知识结构*")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("#### ⚙️ 显示设置")
                
                kg_layout = gr.Radio(
                    choices=["spring", "circular", "kamada_kawai"],
                    value="spring",
                    label="布局算法"
                )
                
                kg_dimension = gr.Radio(
                    choices=["2D", "3D"],
                    value="2D",
                    label="维度"
                )
                
                max_nodes_slider = gr.Slider(
                    minimum=50,
                    maximum=1000,
                    value=300,
                    step=50,
                    label="最大显示节点数"
                )
                
                show_edges_checkbox = gr.Checkbox(
                    label="显示连线",
                    value=True
                )
                
                # 节点类型筛选
                gr.Markdown("#### 🎯 节点类型筛选")
                show_questions = gr.Checkbox(label="题目", value=True)
                show_major_points = gr.Checkbox(label="知识点大类", value=True)
                show_minor_points = gr.Checkbox(label="知识点小类", value=True)
                show_concepts = gr.Checkbox(label="概念", value=True)
                show_methods = gr.Checkbox(label="方法", value=True)
                
                refresh_kg_btn = gr.Button("🔄 刷新图谱", variant="primary", size="lg")
                rebuild_kg_btn = gr.Button("🔨 重建知识图谱", variant="secondary")
            
            with gr.Column(scale=3):
                # 知识图谱显示
                kg_plot = gr.Plot(label="知识图谱", value=None)
                
                # 统计信息
                kg_stats_display = gr.Markdown("### 📊 图谱统计\n\n点击刷新按钮加载图谱")
        
        # 操作状态显示
        operation_status = gr.Markdown("")
        
        # 绑定刷新事件
        refresh_kg_btn.click(
            fn=self._refresh_knowledge_graph_advanced,
            inputs=[
                kg_layout, kg_dimension, max_nodes_slider, show_edges_checkbox,
                show_questions, show_major_points, show_minor_points, 
                show_concepts, show_methods
            ],
            outputs=[kg_plot, kg_stats_display]
        )
        
        # 重建知识图谱
        rebuild_kg_btn.click(
            fn=self._rebuild_knowledge_graph,
            outputs=[operation_status, kg_plot, kg_stats_display]
        )
    
    def _create_question_management_tab(self):
        """创建题库管理标签页(教师)"""
        gr.Markdown("### 📚 题库管理")
        
        # 筛选器
        with gr.Row():
            major_filter = gr.Dropdown(
                label="知识点大类",
                choices=["全部"] + list(self.db.get_knowledge_points().keys()),
                value="全部"
            )
            minor_filter = gr.Dropdown(
                label="知识点小类",
                choices=["全部"],
                value="全部"
            )
        
        search_btn = gr.Button("🔍 查询", variant="primary")
        
        # 题目列表
        questions_table = gr.Dataframe(
            headers=["题号", "问题", "知识点大类", "知识点小类", "难度"],
            interactive=False
        )
        
        # 显示题目总数
        total_count = gr.Markdown("### 📊 题目总数: 0")
        
        # 更新小类选项
        def update_minor_choices(major):
            if major == "全部":
                return gr.update(choices=["全部"])
            kp = self.db.get_knowledge_points()
            minors = kp.get(major, [])
            return gr.update(choices=["全部"] + minors)
        
        major_filter.change(
            fn=update_minor_choices,
            inputs=[major_filter],
            outputs=[minor_filter]
        )
        
        search_btn.click(
            fn=self._search_questions,
            inputs=[major_filter, minor_filter],
            outputs=[questions_table, total_count]
        )
        
        # 添加题目
        with gr.Accordion("➕ 添加新题目", open=False):
            with gr.Row():
                new_q_content = gr.Textbox(label="题目内容", lines=3)
                new_q_answer = gr.Textbox(label="答案", lines=2)
            
            with gr.Row():
                new_q_major = gr.Dropdown(
                    label="知识点大类",
                    choices=list(self.db.get_knowledge_points().keys())
                )
                new_q_minor = gr.Textbox(label="知识点小类")
                new_q_diff = gr.Slider(
                    label="难度",
                    minimum=0,
                    maximum=1,
                    value=0.5,
                    step=0.05
                )
            
            new_q_explanation = gr.Textbox(label="解析", lines=3)
            add_btn = gr.Button("➕ 添加题目", variant="primary")
            add_status = gr.Markdown("")
        
        add_btn.click(
            fn=self._add_question,
            inputs=[
                new_q_content, new_q_answer, new_q_major,
                new_q_minor, new_q_diff, new_q_explanation
            ],
            outputs=[add_status, questions_table, total_count]
        )
    
    def _create_student_management_tab(self):
        """创建学生管理标签页(教师)"""
        gr.Markdown("### 👥 学生管理")
        
        refresh_btn = gr.Button("🔄 刷新列表", variant="primary")
        
        students_table = gr.Dataframe(
            headers=["用户名", "姓名", "注册时间", "最后登录"],
            interactive=False
        )
        
        with gr.Row():
            student_selector = gr.Dropdown(label="选择学生", choices=[])
            view_btn = gr.Button("👁️ 查看详情", variant="primary")
        
        with gr.Column():
            student_detail = gr.Markdown("### 📋 学生详情\n\n请先选择学生")
            student_radar = gr.Plot(label="📊 学生掌握度雷达图")
        
        refresh_btn.click(
            fn=self._load_students_list,
            outputs=[students_table, student_selector]
        )
        
        view_btn.click(
            fn=self._view_student_detail,
            inputs=[student_selector],
            outputs=[student_detail, student_radar]
        )
    
    def _create_system_management_tab(self):
        """创建系统管理标签页"""
        gr.Markdown("### ⚙️ 系统管理")
        
        stats_display = gr.Markdown(self._get_system_stats())
        
        with gr.Row():
            refresh_stats_btn = gr.Button("🔄 刷新统计", variant="primary")
        
        refresh_stats_btn.click(
            fn=self._get_system_stats,
            outputs=[stats_display]
        )
    
    # ==================== 辅助方法 ====================
    
    def _start_assessment_for_current_user(self, num_questions: int):
        """为当前登录用户开始测评"""
        if not self.current_user:
            return None, gr.update(visible=False), "请先登录", ""
        
        student_id = self.current_user['username']
        session = self.system.start_smart_assessment(student_id, int(num_questions))
        
        if session:
            question = session['current_question']
            progress = f"### 进度: {session['current_index']}/{session['total_questions']}"
            
            return (
                session,
                gr.update(visible=True),
                question['问题'],
                progress
            )
        else:
            return None, gr.update(visible=False), "无法开始测评", ""
    
    def _submit_answer(self, session, answer):
        """提交答案 - 修复了图标显示bug"""
        if not session:
            return session, "请先开始测评", gr.update(), gr.update(), ""
        
        session = self.system.submit_answer(session, answer)
        last_result = session['last_result']
        
        # 🔧 修复:根据正确与否选择图标
        result_icon = "✅" if last_result['is_correct'] else "❌"
        result_text = "正确" if last_result['is_correct'] else "错误"
        
        feedback = f"""
### {result_icon} 判定结果: {result_text}

**你的答案:** {answer}

**标准答案:** {last_result['question']['答案']}

**评判理由:** {last_result['check_reason']}

**掌握度变化:** {last_result['mastery_before']:.1%} → {last_result['mastery_after']:.1%} ({last_result['mastery_change']:+.1%})
"""
        
        return (
            session,
            gr.update(value=feedback, visible=True),
            gr.update(visible=False),
            gr.update(visible=True),
            ""
        )
    
    def _restart_assessment(self):
        """重新开始测评 - 修复版"""
        logger.info("🔄 重置测评状态")
        return (
            None,                                      # session_state
            "",                                        # answer_input
            gr.update(visible=True),                   # submit_btn
            gr.update(visible=False),                  # next_btn
            gr.update(value="", visible=False),        # feedback_box
            "### 📊 进度: 0/0",                        # progress_text
            "",                                        # question_text
            ""                                         # report_display
        )

    def _next_question(self, session):
        """下一题 - 修复版:适配新的输出列表"""
        if not session:
            return (
                None,                                      # session_state
                "",                                        # question_text
                "### 📊 进度: 0/0",                        # progress_text
                gr.update(value="", visible=False),        # 🔧 修复:清空并隐藏反馈
                gr.update(visible=True),                   # submit_btn
                gr.update(visible=False),                  # next_btn
                "",                                        # answer_input
                ""                                         # report_display
            )
    
        try:
            # 检查是否完成
            if session['current_index'] >= session['total_questions']:
                logger.info("📊 测评完成,正在生成报告...")
            
                # 使用盘古7B生成报告
                report = self.system.generate_report(session)
                
                return (
                    session,                                   # session_state
                    "",                                        # question_text (清空)
                    f"### 📊 进度: {session['current_index']}/{session['total_questions']} (已完成)",
                    gr.update(value="", visible=False),        # feedback_box (清空并隐藏)
                    gr.update(visible=False),                  # submit_btn (隐藏)
                    gr.update(visible=False),                  # next_btn (隐藏)
                    "",                                        # answer_input (清空)
                    report                                     # report_display (显示报告)
                )
        
            # 加载下一题
            session = self.system.next_question(session)
            question = session['current_question']
            major = session['current_major_point']
            minor = session['current_minor_point']
        
            progress_md = f"### 📊 进度: {session['current_index']}/{session['total_questions']}"
            kp_md = f"**当前知识点:** {major} → {minor}"
            ai_status = "**🤖 AI状态:** 已选择下一题(智能推荐)"
        
            return (
                session,
                question['问题'],
                progress_md,
                gr.update(value="", visible=False),        
                gr.update(visible=True),                   
                gr.update(visible=False),                 
                "",                                       
                ""                                        
            )
        except Exception as e:
            logger.error(f"加载下一题失败: {e}")
            return (
                session, f"错误: {str(e)}", "进度: N/A", 
                gr.update(value="", visible=False),        
                gr.update(visible=True), 
                gr.update(visible=False), "", 
                ""
            )

    def _next_question_fixed(self, session):
        """下一题 - 完整修复版(包含区域可见性控制)"""
        if not session:
            return (
                None,                                    
                "",                                      
                "### 📊 进度: 0/0",                       
                gr.update(value="", visible=False),       
                gr.update(visible=True),                   
                gr.update(visible=False),                
                "",                                       
                gr.update(visible=True),                 
                gr.update(visible=False),                 
                ""                                        
            )
    
        try:
            # 检查是否完成
            if session['current_index'] >= session['total_questions']:
                logger.info("📊 测评完成,正在生成报告...")
            
                # 使用盘古7B生成报告
                report = self.system.generate_report(session)
                
                return (
                    session,                                   # session_state
                    "",                                        # question_text
                    f"### 📊 进度: {session['current_index']}/{session['total_questions']} (已完成)",
                    gr.update(value="", visible=False),        # feedback_box
                    gr.update(visible=False),                  # submit_btn
                    gr.update(visible=False),                  # next_btn
                    "",                                        # answer_input
                    gr.update(visible=False),                 
                    gr.update(visible=True),                  
                    report                                     
                )
        
            # 加载下一题
            session = self.system.next_question(session)
            question = session['current_question']
            major = session['current_major_point']
            minor = session['current_minor_point']
        
            progress_md = f"### 📊 进度: {session['current_index']}/{session['total_questions']}"
        
            return (
                session,
                question['问题'],
                progress_md,
                gr.update(value="", visible=False),        # feedback_box
                gr.update(visible=True),                   # submit_btn
                gr.update(visible=False),                  # next_btn
                "",                                        # answer_input
                gr.update(visible=True),                 
                gr.update(visible=False),                 
                ""                                        
            )
        except Exception as e:
            logger.error(f"加载下一题失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return (
                session, f"错误: {str(e)}", "进度: N/A", 
                gr.update(value="", visible=False),
                gr.update(visible=True), 
                gr.update(visible=False), "",
                gr.update(visible=True),
                gr.update(visible=False),
                ""
            )    
    
    def _load_student_data(self):
        """加载学生数据"""
        if not self.current_user:
            return "请先登录", "", None, []
        
        student_id = self.current_user['username']
        profile = self.db.get_student_profile(student_id)
        
        overall_md = f"""
### 📈 整体学习状况

| 指标 | 数值 |
|------|------|
| 整体掌握度 | **{profile['overall_mastery']:.1%}** |
| 已学知识点 | {profile['total_knowledge_points']} 个 |
| 累计答题数 | {profile['total_answers']} 题 |
"""
        
        weak_points = profile['weak_points']
        if weak_points:
            weak_md = "### ⚠️ 薄弱知识点\n\n"
            for i, (major, minor, mastery) in enumerate(weak_points[:5], 1):
                bar = self._create_mastery_bar(mastery)
                weak_md += f"{i}. **{major} / {minor}**: {bar} {mastery:.1%}\n\n"
        else:
            weak_md = "### ⚠️ 薄弱知识点\n\n✅ 无明显薄弱点"
        
        radar_fig = self._create_radar_chart(profile)
        
        history = self.db.get_answer_history(student_id, limit=20)
        history_data = []
        for h in history:
            history_data.append([
                h['question_no'],
                h['major_point'],
                h['minor_point'],
                "✅" if h['is_correct'] else "❌",
                f"{h['mastery_after'] - h['mastery_before']:+.3f}",
                h['answered_at']
            ])
        
        return overall_md, weak_md, radar_fig, history_data
    
    def _refresh_knowledge_graph(self, layout: str, dimension: str, 
                                 max_nodes: int, show_edges: bool):
        """刷新知识图谱(学生版)"""
        try:
            logger.info(f"🎨 生成知识图谱: {dimension}, 布局={layout}, 节点数≤{max_nodes}")
            
            if dimension == "3D":
                fig = self._create_3d_knowledge_graph(layout, max_nodes, show_edges)
            else:
                fig = self._create_2d_knowledge_graph(layout, max_nodes, show_edges)
            
            # 统计信息
            stats = self.system.visualizer.get_graph_statistics()
            stats_md = f"""
### 📊 知识图谱统计

| 指标 | 数值 |
|------|------|
| 总节点数 | **{stats['total_nodes']}** |
| 总边数 | **{stats['total_edges']}** |
| 图密度 | {stats['density']:.4f} |
| 连通性 | {'✅ 连通' if stats['is_connected'] else '❌ 非连通'} |

#### 节点类型分布
"""
            for node_type, count in stats['node_types'].items():
                type_name = {'knowledge': '知识点', 'difficulty': '难度', 'question': '题目'}.get(node_type, node_type)
                stats_md += f"- **{type_name}**: {count} 个\n"
            
            return fig, stats_md
            
        except Exception as e:
            logger.error(f"❌ 刷新知识图谱失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            error_md = f"### ❌ 错误\n\n刷新知识图谱失败: {str(e)}"
            return None, error_md
    
    def _refresh_knowledge_graph_advanced(self, layout: str, dimension: str,
                                          max_nodes: int, show_edges: bool,
                                          show_questions: bool, show_major: bool,
                                          show_minor: bool, show_concepts: bool,
                                          show_methods: bool):
        """刷新知识图谱(教师版 - 支持节点筛选)"""
        try:
            logger.info(f"🎨 生成知识图谱(高级): {dimension}, 布局={layout}")
            
            # 构建节点类型筛选列表
            node_types_filter = []
            if show_questions:
                node_types_filter.append('question')
            if show_major:
                node_types_filter.append('major_point')
            if show_minor:
                node_types_filter.append('minor_point')
            if show_concepts:
                node_types_filter.append('concept')
            if show_methods:
                node_types_filter.append('method')
            
            if dimension == "3D":
                fig = self._create_3d_knowledge_graph(layout, max_nodes, show_edges, node_types_filter)
            else:
                fig = self._create_2d_knowledge_graph(layout, max_nodes, show_edges, node_types_filter)
            
            # 统计信息
            stats = self.system.visualizer.get_graph_statistics()
            stats_md = f"""
### 📊 知识图谱统计

| 指标 | 数值 |
|------|------|
| 总节点数 | **{stats['total_nodes']}** |
| 显示节点 | **≤{max_nodes}** |
| 总边数 | **{stats['total_edges']}** |
| 图密度 | {stats['density']:.4f} |

#### 节点类型分布
"""
            for node_type, count in stats['node_types'].items():
                type_name = {
                    'question': '题目',
                    'major_point': '知识点大类',
                    'minor_point': '知识点小类',
                    'concept': '概念',
                    'method': '方法'
                }.get(node_type, node_type)
                stats_md += f"- **{type_name}**: {count} 个\n"
            
            return fig, stats_md
            
        except Exception as e:
            logger.error(f"❌ 刷新知识图谱失败: {e}")
            error_md = f"### ❌ 错误\n\n{str(e)}"
            return None, error_md
    
    def _create_2d_knowledge_graph(self, layout: str, max_nodes: int, 
                                   show_edges: bool, node_types_filter: list = None):
        """创建2D知识图谱"""
        import networkx as nx
        import random
        
        graph = self.system.knowledge_graph
        
        # 节点采样
        if graph.number_of_nodes() > max_nodes:
            logger.info(f"⚠️  节点数 {graph.number_of_nodes()} 超过限制 {max_nodes},进行采样...")
            
            # 保留重要节点
            if node_types_filter:
                important_nodes = [n for n, d in graph.nodes(data=True) 
                                 if d.get('type') in node_types_filter]
            else:
                important_nodes = [n for n, d in graph.nodes(data=True) 
                                 if d.get('type') in ['major_point', 'minor_point', 'concept', 'method']]
            
            question_nodes = [n for n, d in graph.nodes(data=True) 
                            if d.get('type') == 'question']
            
            remaining = max_nodes - len(important_nodes)
            if remaining > 0:
                sampled = important_nodes + random.sample(question_nodes, min(remaining, len(question_nodes)))
            else:
                sampled = important_nodes[:max_nodes]
            
            graph = graph.subgraph(sampled).copy()
        
        # 计算布局
        if layout == 'spring':
            pos = nx.spring_layout(graph, k=2, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(graph)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(graph)
        else:
            pos = nx.spring_layout(graph)
        
        # 创建Plotly图形
        edge_traces = []
        if show_edges:
            for u, v in graph.edges():
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                edge_traces.append(go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode='lines',
                    line=dict(width=1, color='#888'),
                    hoverinfo='none',
                    showlegend=False
                ))
        
        # 按类型分组节点
        node_colors = {
            'question': '#95E1D3',
            'major_point': '#FF6B6B',
            'minor_point': '#4ECDC4',
            'concept': '#FFD93D',
            'method': '#A8E6CF',
            'default': '#CCCCCC'
        }
        
        node_sizes = {
            'question': 8,
            'major_point': 25,
            'minor_point': 18,
            'concept': 15,
            'method': 12,
            'default': 10
        }
        
        type_names = {
            'question': '题目',
            'major_point': '知识点大类',
            'minor_point': '知识点小类',
            'concept': '概念',
            'method': '方法'
        }
        
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
        
        for node_type, group_data in node_groups.items():
            if not group_data['x']:
                continue
            
            node_traces.append(go.Scatter(
                x=group_data['x'],
                y=group_data['y'],
                mode='markers+text',
                marker=dict(
                    size=node_sizes.get(node_type, 10),
                    color=node_colors.get(node_type, '#CCCCCC'),
                    line=dict(color='white', width=1)
                ),
                text=group_data['texts'],
                textposition='top center',
                textfont=dict(size=10),
                hoverinfo='text',
                name=type_names.get(node_type, node_type),
                showlegend=True
            ))
        
        fig = go.Figure(data=edge_traces + node_traces)
        
        fig.update_layout(
            title=dict(text=f"知识图谱 2D 可视化<br><sub>节点: {graph.number_of_nodes()} | 边: {graph.number_of_edges()}</sub>", 
                      x=0.5, xanchor='center'),
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            height=700
        )
        
        return fig
    
    def _create_3d_knowledge_graph(self, layout: str, max_nodes: int,
                                   show_edges: bool, node_types_filter: list = None):
        """创建3D知识图谱"""
        import networkx as nx
        import random
        
        graph = self.system.knowledge_graph
        
        # 节点采样
        if graph.number_of_nodes() > max_nodes:
            if node_types_filter:
                important_nodes = [n for n, d in graph.nodes(data=True) 
                                 if d.get('type') in node_types_filter]
            else:
                important_nodes = [n for n, d in graph.nodes(data=True) 
                                 if d.get('type') in ['major_point', 'minor_point', 'concept', 'method']]
            
            question_nodes = [n for n, d in graph.nodes(data=True) 
                            if d.get('type') == 'question']
            
            remaining = max_nodes - len(important_nodes)
            if remaining > 0:
                sampled = important_nodes + random.sample(question_nodes, min(remaining, len(question_nodes)))
            else:
                sampled = important_nodes[:max_nodes]
            
            graph = graph.subgraph(sampled).copy()
        
        # 计算3D布局
        pos = nx.spring_layout(graph, dim=3, k=0.5, iterations=50)
        
        # 创建边
        edge_traces = []
        if show_edges:
            edge_x, edge_y, edge_z = [], [], []
            for u, v in graph.edges():
                x0, y0, z0 = pos[u]
                x1, y1, z1 = pos[v]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                edge_z.extend([z0, z1, None])
            
            edge_traces.append(go.Scatter3d(
                x=edge_x, y=edge_y, z=edge_z,
                mode='lines',
                line=dict(color='#888', width=1),
                hoverinfo='none',
                showlegend=False
            ))
        
        # 创建节点
        node_colors = {
            'question': '#95E1D3',
            'major_point': '#FF6B6B',
            'minor_point': '#4ECDC4',
            'concept': '#FFD93D',
            'method': '#A8E6CF',
            'default': '#CCCCCC'
        }
        
        node_sizes = {
            'question': 8,
            'major_point': 25,
            'minor_point': 18,
            'concept': 15,
            'method': 12,
            'default': 10
        }
        
        type_names = {
            'question': '题目',
            'major_point': '知识点大类',
            'minor_point': '知识点小类',
            'concept': '概念',
            'method': '方法'
        }
        
        node_traces = []
        node_groups = {}
        
        for node, data in graph.nodes(data=True):
            node_type = data.get('type', 'default')
            if node_type not in node_groups:
                node_groups[node_type] = {'nodes': [], 'texts': []}
            
            x, y, z = pos[node]
            node_groups[node_type]['nodes'].append((x, y, z))
            
            name = data.get('name', node)
            if len(str(name)) > 30:
                name = str(name)[:27] + "..."
            node_groups[node_type]['texts'].append(name)
        
        for node_type, group_data in node_groups.items():
            if not group_data['nodes']:
                continue
            
            x_vals, y_vals, z_vals = zip(*group_data['nodes'])
            
            node_traces.append(go.Scatter3d(
                x=x_vals, y=y_vals, z=z_vals,
                mode='markers+text',
                marker=dict(
                    size=node_sizes.get(node_type, 10),
                    color=node_colors.get(node_type, '#CCCCCC'),
                    line=dict(color='white', width=0.5)
                ),
                text=group_data['texts'],
                textposition='top center',
                textfont=dict(size=8),
                hoverinfo='text',
                name=type_names.get(node_type, node_type),
                showlegend=True
            ))
        
        fig = go.Figure(data=edge_traces + node_traces)
        
        fig.update_layout(
            title=dict(text=f"知识图谱 3D 可视化<br><sub>节点: {graph.number_of_nodes()} | 边: {graph.number_of_edges()}</sub>",
                      x=0.5, xanchor='center'),
            showlegend=True,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=40),
            scene=dict(
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                zaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                bgcolor='white'
            ),
            height=700
        )
        
        return fig
    
    def _rebuild_knowledge_graph(self):
        """重建知识图谱"""
        try:
            logger.info("🔨 开始重建知识图谱...")
            
            if hasattr(self.system, 'force_rebuild_kg'):
                success = self.system.force_rebuild_kg()
                
                if success:
                    # 重新加载图谱
                    fig = self._create_2d_knowledge_graph('spring', 300, True)
                    stats = self.system.visualizer.get_graph_statistics()
                    stats_md = f"""
### ✅ 知识图谱重建成功!

| 指标 | 数值 |
|------|------|
| 节点数 | **{stats['total_nodes']}** |
| 边数 | **{stats['total_edges']}** |
"""
                    
                    return "✅ 知识图谱重建成功!", fig, stats_md
                else:
                    return "❌ 知识图谱重建失败,请查看日志", None, "### ❌ 重建失败"
            else:
                return "❌ 系统不支持知识图谱重建", None, "### ❌ 不支持"
                
        except Exception as e:
            logger.error(f"❌ 重建知识图谱失败: {e}")
            return f"❌ 重建失败: {str(e)}", None, f"### ❌ 错误\n\n{str(e)}"
    
    def _create_radar_chart(self, profile: Dict[str, Any]) -> go.Figure:
        """创建雷达图"""
        knowledge_points = profile.get('knowledge_points', {})
        
        if not knowledge_points:
            fig = go.Figure()
            fig.add_annotation(text="暂无数据", x=0.5, y=0.5, showarrow=False)
            return fig
        
        categories = []
        values = []
        for major, minors in knowledge_points.items():
            if isinstance(minors, dict):
                avg_mastery = sum(minors.values()) / len(minors)
            else:
                avg_mastery = minors
            categories.append(major)
            values.append(avg_mastery)
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='掌握度'
        ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=False,
            height=400
        )
        
        return fig
    
    def _search_questions(self, major: str, minor: str):
        """搜索题目"""
        major_filter = None if major == "全部" else major
        minor_filter = None if minor == "全部" else minor
        
        questions = self.db.get_questions_filtered(
            major_point=major_filter,
            minor_point=minor_filter
        )
        
        table_data = []
        for q in questions:
            table_data.append([
                q['题号'],
                q['问题'][:50] + "..." if len(q['问题']) > 50 else q['问题'],
                q['知识点大类'],
                q['知识点小类'],
                f"{q['难度']:.2f}"
            ])
        
        total_count_md = f"### 📊 题目总数: {len(questions)}"
        
        return table_data, total_count_md
    
    def _add_question(self, content, answer, major, minor, difficulty, explanation):
        """添加题目"""
        all_q = self.db.get_all_questions()
        max_no = max([q['题号'] for q in all_q]) if all_q else 0
        
        question_data = {
            '题号': max_no + 1,
            '问题': content,
            '答案': answer,
            '知识点大类': major,
            '知识点小类': minor,
            '难度': difficulty,
            '解析': explanation
        }
        
        if self.db.insert_question(question_data):
            new_table, total_count = self._search_questions("全部", "全部")
            return "✅ 题目添加成功!", new_table, total_count
        else:
            return "❌ 添加失败", [], "### 📊 题目总数: 0"
    
    def _load_students_list(self):
        """加载学生列表"""
        students = self.db.get_all_students()
        
        table_data = []
        choices = []
        for s in students:
            table_data.append([
                s['username'],
                s['real_name'] or '',
                s['created_at'],
                s['last_login'] or '未登录'
            ])
            choices.append(s['username'])
        
        return table_data, gr.update(choices=choices)
    
    def _view_student_detail(self, student_id: str):
        """查看学生详情"""
        if not student_id:
            return "请选择学生", None
        
        profile = self.db.get_student_profile(student_id)
        
        detail_md = f"""
### 📊 学生档案: {student_id}

**整体掌握度:** {profile['overall_mastery']:.1%}  
**已学知识点:** {profile['total_knowledge_points']} 个  
**累计答题:** {profile['total_answers']} 题

#### ⚠️ 薄弱知识点
"""
        for major, minor, mastery in profile['weak_points'][:5]:
            detail_md += f"- {major} / {minor}: {mastery:.1%}\n"
        
        radar_fig = self._create_radar_chart(profile)
        
        return detail_md, radar_fig
    
    def _get_system_stats(self) -> str:
        """获取系统统计"""
        q_stats = self.db.get_question_statistics()
        students = self.db.get_all_students()
        
        stats_md = f"""
### 📊 系统统计

#### 题库信息
- **总题目数:** {q_stats['总题目数']}
- **知识点大类:** {len(q_stats['知识点大类分布'])}
- **知识点小类:** {len(q_stats['知识点小类分布'])}

#### 难度分布
- 简单: {q_stats['难度分布']['简单']}
- 中等: {q_stats['难度分布']['中等']}
- 困难: {q_stats['难度分布']['困难']}

#### 用户信息
- **学生数量:** {len(students)}
"""
        
        return stats_md
    
    def _create_mastery_bar(self, mastery: float, length: int = 20) -> str:
        """创建掌握度可视化条"""
        filled = int(mastery * length)
        empty = length - filled
        
        if mastery >= 0.7:
            bar = '🟩' * filled + '⬜' * empty
        elif mastery >= 0.4:
            bar = '🟨' * filled + '⬜' * empty
        else:
            bar = '🟥' * filled + '⬜' * empty
        
        return bar


def create_enhanced_ui(system_core, db_manager) -> gr.Blocks:
    """创建增强版UI"""
    ui = EnhancedEducationUI(system_core, db_manager)
    return ui.create_interface()