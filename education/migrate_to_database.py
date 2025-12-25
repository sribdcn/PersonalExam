
"""
数据迁移脚本
将JSON数据迁移到SQLite数据库
"""

import sys
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from database import create_database_manager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """主函数"""
    print("=" * 70)
    print("开始数据迁移：JSON → SQLite")
    print("=" * 70)
    
    questions_file = PROJECT_ROOT / "data" / "question_database_4.json"
    states_file = PROJECT_ROOT / "data" / "student_states.json"
    db_path = PROJECT_ROOT / "data" / "education_system.db"
    
    if not questions_file.exists():
        logger.error(f"题库文件不存在: {questions_file}")
        return
    
    if not states_file.exists():
        logger.warning(f"学生状态文件不存在: {states_file}")
        logger.info("将只迁移题库数据")
    
    try:
        logger.info("初始化数据库...")
        db = create_database_manager(str(db_path))
        
        logger.info("创建默认用户...")
        db.create_user("student_001", "123456", "student", "学生001")
        db.create_user("teacher", "admin123", "teacher", "管理员")
        logger.info("默认用户创建完成")
        logger.info("   学生账号: student_001 / 123456")
        logger.info("   教师账号: teacher / admin123")
        
        logger.info("\n" + "="*70)
        logger.info("开始迁移数据...")
        logger.info("="*70)
        
        stats = db.migrate_from_json(
            str(questions_file),
            str(states_file) if states_file.exists() else None
        )
        
        print("\n" + "=" * 70)
        print("数据迁移完成！")
        print("=" * 70)
        print(f"迁移统计:")
        print(f"   - 题目: {stats['questions']} 道")
        print(f"   - 学生状态: {stats['states']} 条")
        print(f"   - 答题历史: {stats['history']} 条")
        print(f"\n数据库文件: {db_path}")
        print("=" * 70)
        
        logger.info("\n🔍 验证迁移结果...")
        question_stats = db.get_question_statistics()
        print(f"\n题库验证:")
        print(f"   - 总题目: {question_stats['总题目数']}")
        print(f"   - 知识点大类: {len(question_stats['知识点大类分布'])}")
        print(f"   - 知识点小类: {len(question_stats['知识点小类分布'])}")
        
        students = db.get_all_students()
        print(f"\n👥 学生数量: {len(students)}")
        
        print("\n迁移成功！可以开始使用新系统了。")
        
    except Exception as e:
        logger.error(f"迁移失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()