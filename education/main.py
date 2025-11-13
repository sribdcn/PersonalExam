# -*- coding: utf-8 -*-
"""
Copyright (c) 2025 AI系统与应用课题组@SRIBD

基于LLM和知识图谱协同的个性化出题系统 (PersonalExam)
Personalized Question Generation System Based on LLM and Knowledge Graph Collaboration
"""

import sys
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from system_core import create_system_core
from ui.main_ui import create_ui


def setup_logging():
    """配置日志系统"""
    log_config = config.LOGGING_CONFIG
    log_file = Path(log_config['log_file'])
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, log_config['level']),
        format=log_config['format'],
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding='utf-8')
        ]
    )


def main():
    print("=" * 70)
    print("🧠 基于LLM和知识图谱协同的个性化出题系统")
    print("=" * 70)
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 系统启动中...")
    
    try:
        # 检查模型文件
        import os
        if not os.path.exists(config.PANGU_MODEL_PATH):
            logger.error(f"❌ 盘古7B模型文件不存在: {config.PANGU_MODEL_PATH}")
            print(f"\n❌ 错误: 模型文件不存在")
            print(f"   模型路径: {config.PANGU_MODEL_PATH}")
            print("   请确保模型文件已正确放置")
            sys.exit(1)
        
        logger.info("✅ 检测到盘古7B模型文件")
        
        # 检查题库文件
        if not config.QUESTION_DB.exists():
            logger.error(f"❌ 题库文件不存在: {config.QUESTION_DB}")
            print(f"\n❌ 错误: 题库文件不存在")
            print(f"   题库路径: {config.QUESTION_DB}")
            print("   请确保 question_database_2.json 已放置在 data 目录")
            sys.exit(1)
        
        logger.info("✅ 检测到题库文件")
        
        print("\n✅ 系统将使用以下配置:")
        print(f"   - 模型: {config.SYSTEM_INFO['model']}")
        print(f"   - 设备: {config.SYSTEM_INFO['device']}")
        print(f"   - 题库: {config.QUESTION_DB.name}")
        print()
        
        # 初始化系统核心
        logger.info("⚙️  正在初始化智能系统核心...")
        system_core = create_system_core(config)
        
        logger.info("✅ 系统核心初始化完成")
        
        # 显示题库信息
        stats = system_core.get_database_statistics()
        logger.info(f"📚 题库统计: 总题目 {stats['总题目数']}")
        logger.info(f"📊 知识点大类: {len(stats['知识点大类分布'])} 个")
        logger.info(f"📋 知识点小类: {len(stats['知识点小类分布'])} 个")
        
        # 创建UI界面
        logger.info("🎨 正在创建UI界面...")
        interface = create_ui(system_core)
        
        # 启动服务
        logger.info("✅ 系统启动成功!")
        print("\n" + "=" * 70)
        print("🚀 智能教育系统已启动!")
        print("=" * 70)
        print(f"\n📊 题库信息:")
        print(f"   - 总题目数: {stats['总题目数']}")
        print(f"   - 知识点大类: {len(stats['知识点大类分布'])} 个")
        print(f"   - 知识点小类: {len(stats['知识点小类分布'])} 个")
        
        print(f"\n🤖 模型信息:")
        print(f"   - 模型: {config.SYSTEM_INFO['model']}")
        print(f"   - 设备: {config.SYSTEM_INFO['device']}")
        if system_core.pangu_model:
            npu_count = len(system_core.pangu_model.devices)
            print(f"   - NPU数量: {npu_count}")
        
        print(f"\n🌐 访问地址: http://localhost:{config.UI_CONFIG['port']}")
        print("=" * 70)
        
        print("\n🎯 系统特点:")
        print("   ✅ 细粒度知识点追踪（支持知识点小类）")
        print("   ✅ 自动识别薄弱知识点")
        print("   ✅ 智能推荐学习路径")
        print("   ✅ 实时自适应难度调整")
        print("   ✅ AI驱动个性化出题")
        
        print("\n💡 使用说明:")
        print("   1. 输入学生ID（如 student_001）")
        print("   2. 选择题目数量")
        print("   3. 点击'开始智能测评'")
        print("   4. 系统会自动分析并选择最适合的题目")
        
        print("\n按 Ctrl+C 退出系统\n")
        
        interface.launch(
            server_port=config.UI_CONFIG['port'],
            share=config.UI_CONFIG['share'],
            inbrowser=True,
            server_name="0.0.0.0"
        )
        
    except KeyboardInterrupt:
        logger.info("⏹️  收到退出信号...")
        print("\n\n🛑 系统正在关闭...")
    except Exception as e:
        logger.error(f"❌ 系统运行出错: {e}", exc_info=True)
        print(f"\n❌ 错误: {e}")
        print("详细错误信息请查看日志文件")
        sys.exit(1)
    finally:
        logger.info("👋 系统已关闭")
        print("再见!")


if __name__ == "__main__":
    main()