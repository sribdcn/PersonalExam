# -*- coding: utf-8 -*-
"""
数据库版本主程序 - 带登录系统的教育评估系统
"""

import sys
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from database import create_database_manager
from bkt_database_adapter import create_bkt_database_adapter
from system_core_db import create_system_core_with_db
from enhanced_main_ui import create_enhanced_ui


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


def initialize_default_users(db_manager):
    """初始化默认用户（仅在不存在时创建）"""
    logger = logging.getLogger(__name__)
    
    existing_student = db_manager.verify_user("student_001", "123456")
    if not existing_student:
        if db_manager.create_user("student_001", "123456", "student", "学生001"):
            logger.info("✅ 创建默认学生账号: student_001 / 123456")
    else:
        logger.info("ℹ️  默认学生账号已存在: student_001")
    
    existing_teacher = db_manager.verify_user("teacher", "admin123")
    if not existing_teacher:
        if db_manager.create_user("teacher", "admin123", "teacher", "管理员"):
            logger.info("✅ 创建默认教师账号: teacher / admin123")
    else:
        logger.info("ℹ️  默认教师账号已存在: teacher")


def main():
    print("=" * 70)
    print("🧠 智能教育系统 - 数据库版本（带登录功能）")
    print("=" * 70)
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 系统启动中...")
    
    try:
        logger.info("📦 初始化数据库...")
        db_path = config.DATA_DIR / "education_system.db"
        db_manager = create_database_manager(str(db_path))
        logger.info(f"✅ 数据库已连接: {db_path}")
        
        logger.info("👥 检查并初始化默认用户...")
        initialize_default_users(db_manager)
        
        logger.info("🧠 初始化BKT算法（数据库版）...")
        bkt_algorithm = create_bkt_database_adapter(db_manager)
        
        logger.info("⚙️  初始化系统核心（数据库版）...")
        system_core = create_system_core_with_db(config, db_manager, bkt_algorithm)
        logger.info("✅ 系统核心初始化完成")
        
        stats = db_manager.get_question_statistics()
        logger.info(f"📚 题库统计: 总题目 {stats['总题目数']}")
        logger.info(f"📊 知识点大类: {len(stats['知识点大类分布'])} 个")
        logger.info(f"📋 知识点小类: {len(stats['知识点小类分布'])} 个")
        
        students = db_manager.get_all_students()
        logger.info(f"👥 学生数量: {len(students)}")
        
        logger.info("🎨 创建UI界面（带登录和注册系统）...")
        interface = create_enhanced_ui(system_core, db_manager)
        
        logger.info("✅ 系统启动成功!")
        print("\n" + "=" * 70)
        print("🚀 智能教育系统已启动（数据库版）!")
        print("=" * 70)
        
        print(f"\n📊 数据库信息:")
        print(f"   - 数据库文件: {db_path}")
        print(f"   - 总题目数: {stats['总题目数']}")
        print(f"   - 知识点大类: {len(stats['知识点大类分布'])} 个")
        print(f"   - 知识点小类: {len(stats['知识点小类分布'])} 个")
        print(f"   - 学生数量: {len(students)}")
        
        print(f"\n👥 默认账号:")
        print(f"   学生账号: student_001 / 123456")
        print(f"   教师账号: teacher / admin123")
        
        print(f"\n🌐 访问地址: http://localhost:{config.UI_CONFIG['port']}")
        print("=" * 70)
        
        print("\n🎯 系统特点:")
        print("   ✅ 用户登录和注册系统（学生/教师分离）")
        print("   ✅ 数据库存储（用户、题目、答题记录）")
        print("   ✅ 学生功能：智能测评、学习分析")
        print("   ✅ 教师功能：题库管理、学生管理、数据查看")
        print("   ✅ 细粒度知识点追踪（BKT算法）")
        print("   ✅ 实时自适应难度调整")
        
        print("\n💡 使用说明:")
        print("   1. 在登录界面输入账号密码或点击注册")
        print("   2. 学生登录后可进行测评和查看学习数据")
        print("   3. 教师登录后可管理题库和查看学生情况")
        print("   4. 系统自动保存所有数据到SQLite数据库")
        
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