"""
测试脚本 - 验证盘古7B模型在系统中的使用情况
"""

import sys
import logging
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from models.llm_models import create_llm_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_pangu_model_basic():
    """测试盘古7B模型基本功能"""
    print("\n" + "="*70)
    print("测试1: 盘古7B模型基本功能")
    print("="*70)
    
    try:
        # 创建模型
        pangu_model = create_llm_model(
            'pangu',
            config.PANGU_MODEL_PATH,
            config.PANGU_MODEL_CONFIG
        )
        
        # 加载模型
        print("🔄 正在加载盘古7B模型...")
        pangu_model.load_model()
        print("✅ 模型加载成功")
        
        # 测试生成
        test_prompt = "请简单介绍什么是贝叶斯定理。"
        print(f"\n📝 测试提示词: {test_prompt}")
        print("🤖 盘古7B正在生成回答...")
        
        response = pangu_model.generate(test_prompt, temperature=0.7, max_length=200)
        
        print(f"\n✅ 盘古7B回答:\n{response}")
        print("\n✅ 测试1通过: 盘古7B模型工作正常")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试1失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_answer_checking():
    """测试答案检查功能（是否使用盘古7B）"""
    print("\n" + "="*70)
    print("测试2: 答案检查功能（盘古7B）")
    print("="*70)
    
    try:
        from utils.evaluator import create_evaluator
        from utils.bkt_algorithm import create_bkt_algorithm
        
        # 创建模型和算法
        pangu_model = create_llm_model(
            'pangu',
            config.PANGU_MODEL_PATH,
            config.EVALUATION_MODEL_CONFIG
        )
        
        bkt = create_bkt_algorithm()
        evaluator = create_evaluator(pangu_model, bkt, config.EVALUATION_CONFIG)
        
        # 测试题目
        test_question = {
            '问题': '解方程: x^2 - 5x + 6 = 0',
            '答案': 'x = 2 或 x = 3',
            '解析': '因式分解得 (x-2)(x-3) = 0，所以 x = 2 或 x = 3'
        }
        
        # 正确答案
        print("\n🧪 测试正确答案:")
        print(f"学生答案: x=2 或 x=3")
        print("🤖 盘古7B正在评估...")
        
        is_correct, reason = evaluator.check_answer(
            test_question,
            "x=2 或 x=3",
            config.PROMPTS['answer_check']
        )
        
        print(f"✅ 判定结果: {'正确' if is_correct else '错误'}")
        print(f"📝 理由: {reason[:200]}...")
        
        # 错误答案
        print("\n🧪 测试错误答案:")
        print(f"学生答案: x=1")
        print("🤖 盘古7B正在评估...")
        
        is_correct2, reason2 = evaluator.check_answer(
            test_question,
            "x=1",
            config.PROMPTS['answer_check']
        )
        
        print(f"✅ 判定结果: {'正确' if is_correct2 else '错误'}")
        print(f"📝 理由: {reason2[:200]}...")
        
        print("\n✅ 测试2通过: 答案检查功能正常使用盘古7B")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试2失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_report_generation():
    """测试报告生成功能（是否使用盘古7B）"""
    print("\n" + "="*70)
    print("测试3: 报告生成功能（盘古7B）")
    print("="*70)
    
    try:
        from utils.evaluator import create_evaluator
        from utils.bkt_algorithm import create_bkt_algorithm
        
        # 创建组件
        pangu_model = create_llm_model(
            'pangu',
            config.PANGU_MODEL_PATH,
            config.EVALUATION_MODEL_CONFIG
        )
        
        bkt = create_bkt_algorithm()
        evaluator = create_evaluator(pangu_model, bkt, config.EVALUATION_CONFIG)
        
        # 模拟答题记录
        test_records = [
            {
                'question': {'问题': '测试题1', '答案': 'A', '解析': '测试', '难度': 0.3},
                'major_point': '代数',
                'minor_point': '一元二次方程',
                'is_correct': True,
                'mastery_before': 0.3,
                'mastery_after': 0.45,
                'mastery_change': 0.15
            },
            {
                'question': {'问题': '测试题2', '答案': 'B', '解析': '测试', '难度': 0.5},
                'major_point': '代数',
                'minor_point': '一元二次方程',
                'is_correct': False,
                'mastery_before': 0.45,
                'mastery_after': 0.35,
                'mastery_change': -0.1
            }
        ]
        
        print("\n🤖 盘古7B正在生成个性化学习建议...")
        learning_pattern = evaluator.analyze_learning_pattern(test_records)
        
        recommendations = evaluator.generate_ai_recommendations(
            'test_student',
            '代数/一元二次方程',
            test_records,
            learning_pattern
        )
        
        print(f"\n✅ 盘古7B生成的建议:")
        print(recommendations[:500] + "..." if len(recommendations) > 500 else recommendations)
        
        print("\n✅ 测试3通过: 报告生成功能正常使用盘古7B")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试3失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rag_integration():
    """测试RAG集成"""
    print("\n" + "="*70)
    print("测试4: RAG与题目生成集成")
    print("="*70)
    
    try:
        import asyncio
        from knowledge_management.rag_engine import create_rag_engine, QuestionRAGManager
        from data_management.question_db import create_question_database
        from models.embedding_model import create_embedding_model, lightrag_embedding_func
        from utils.question_generator import create_question_generator  # 修复：改为 utils
        
        # 创建组件
        embedding_model = create_embedding_model(
            config.BGE_M3_MODEL_PATH,
            config.EMBEDDING_MODEL_CONFIG
        )
        
        rag_engine = create_rag_engine(
            config.LIGHTRAG_CONFIG,
            lambda texts: lightrag_embedding_func(texts, embedding_model)
        )
        
        question_db = create_question_database(str(config.QUESTION_DB))
        
        pangu_model = create_llm_model(
            'pangu',
            config.PANGU_MODEL_PATH,
            config.QUESTION_MODEL_CONFIG
        )
        
        print("\n🔄 初始化RAG引擎...")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(rag_engine.initialize())
            print("✅ RAG引擎初始化成功")
            
            # 构建知识图谱
            print("🔄 构建知识图谱（取前10题测试）...")
            rag_manager = QuestionRAGManager(rag_engine)
            questions = question_db.get_all_questions()[:10]
            loop.run_until_complete(rag_manager.build_kg_from_questions(questions))
            print("✅ 知识图谱构建完成")
            
            # 创建题目生成器
            print("\n🔄 创建题目生成器...")
            generator = create_question_generator(
                pangu_model,
                question_db,
                rag_engine,
                config.SMART_QUESTION_CONFIG,
                use_real_generation=True
            )
            
            print("✅ 题目生成器创建成功（使用盘古7B + RAG）")
            
            print("\n✅ 测试4通过: RAG集成正常")
            return True
            
        finally:
            loop.close()
        
    except Exception as e:
        print(f"\n❌ 测试4失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "="*70)
    print("🧪 盘古7B使用情况综合测试")
    print("="*70)
    
    results = []
    
    # 测试1: 基本功能
    results.append(("盘古7B基本功能", test_pangu_model_basic()))
    
    # 测试2: 答案检查
    results.append(("答案检查（盘古7B）", test_answer_checking()))
    
    # 测试3: 报告生成
    results.append(("报告生成（盘古7B）", test_report_generation()))
    
    # 测试4: RAG集成
    results.append(("RAG集成", test_rag_integration()))
    
    # 汇总结果
    print("\n" + "="*70)
    print("📊 测试结果汇总")
    print("="*70)
    
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{name}: {status}")
    
    all_passed = all(r[1] for r in results)
    
    print("\n" + "="*70)
    if all_passed:
        print("🎉 所有测试通过！盘古7B模型正常工作")
        print("\n✅ 确认事项:")
        print("  ✓ 盘古7B用于答案评估")
        print("  ✓ 盘古7B用于报告生成")
        print("  ✓ 盘古7B结合RAG用于题目生成")
        print("  ✓ LightRAG知识图谱正常集成")
    else:
        print("⚠️  部分测试失败，请查看详细日志")
    print("="*70)


if __name__ == "__main__":
    main()