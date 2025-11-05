"""
评估器测试脚本
用于测试优化后的答案检查功能
"""

import sys
import logging
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from models import create_llm_model
from utils.evaluator import create_evaluator

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_answer_checking():
    """测试答案检查功能"""
    
    print("\n" + "="*60)
    print("答案检查功能测试")
    print("="*60 + "\n")
    
    # 创建模型和评估器
    logger.info("正在初始化模型和评估器...")
    pangu_model = create_llm_model(
        'pangu',
        config.PANGU_MODEL_PATH,
        config.EVALUATION_MODEL_CONFIG
    )
    
    evaluator = create_evaluator(pangu_model, config.EVALUATION_CONFIG)
    
    # 测试用例
    test_cases = [
        {
            'name': '测试1：单调性问题 - 不完整答案',
            'question': {
                '问题': '设函数 g(x) = ln x − (x − 1)/x，定义域 x>0。判断 g(x) 的单调性。',
                '答案': 'g(x) 在 (0,1) 上单调减少，在 (1, +∞) 上单调增加。',
                '解析': 'g′(x) = 1/x − 1/x^2 = (x − 1)/x^2。当 x>1 时 g′>0；当 0<x<1 时 g′<0。',
                '难度': '简单',
                '知识点': '代数'
            },
            'student_answer': '单调递增',
            'expected': False  # 期望判定为错误
        },
        {
            'name': '测试2：单调性问题 - 完整答案',
            'question': {
                '问题': '设函数 g(x) = ln x − (x − 1)/x，定义域 x>0。判断 g(x) 的单调性。',
                '答案': 'g(x) 在 (0,1) 上单调减少，在 (1, +∞) 上单调增加。',
                '解析': 'g′(x) = 1/x − 1/x^2 = (x − 1)/x^2。当 x>1 时 g′>0；当 0<x<1 时 g′<0。',
                '难度': '简单',
                '知识点': '代数'
            },
            'student_answer': '在 (0,1) 上单调减少，在 (1,+∞) 上单调增加',
            'expected': True  # 期望判定为正确
        },
        {
            'name': '测试3：方程求解 - 遗漏部分解',
            'question': {
                '问题': '解方程 x^2 - 5x + 6 = 0',
                '答案': 'x = 2 或 x = 3',
                '解析': '因式分解: (x-2)(x-3) = 0',
                '难度': '简单',
                '知识点': '代数'
            },
            'student_answer': 'x = 2',
            'expected': False  # 期望判定为错误（遗漏了x=3）
        },
        {
            'name': '测试4：方程求解 - 完整答案',
            'question': {
                '问题': '解方程 x^2 - 5x + 6 = 0',
                '答案': 'x = 2 或 x = 3',
                '解析': '因式分解: (x-2)(x-3) = 0',
                '难度': '简单',
                '知识点': '代数'
            },
            'student_answer': 'x = 2 或 x = 3',
            'expected': True  # 期望判定为正确
        },
        {
            'name': '测试5：区间问题 - 遗漏区间',
            'question': {
                '问题': '解不等式 (x-2)(x-3) ≤ 0',
                '答案': '2 ≤ x ≤ 3',
                '解析': '当 2 ≤ x ≤ 3 时，两因子异号或为零',
                '难度': '中等',
                '知识点': '代数'
            },
            'student_answer': 'x ≥ 2',
            'expected': False  # 期望判定为错误（遗漏了上界）
        },
        {
            'name': '测试6：简洁表述',
            'question': {
                '问题': '设函数 g(x) = ln x − (x − 1)/x，定义域 x>0。判断 g(x) 的单调性。',
                '答案': 'g(x) 在 (0,1) 上单调减少，在 (1, +∞) 上单调增加。',
                '解析': 'g′(x) = (x − 1)/x^2',
                '难度': '简单',
                '知识点': '代数'
            },
            'student_answer': '(0,1)减，(1,+∞)增',
            'expected': True  # 期望判定为正确（信息完整，表述简洁）
        }
    ]
    
    # 运行测试
    passed = 0
    failed = 0
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"{test_case['name']}")
        print(f"{'='*60}")
        print(f"题目: {test_case['question']['问题']}")
        print(f"标准答案: {test_case['question']['答案']}")
        print(f"学生答案: {test_case['student_answer']}")
        print(f"期望结果: {'正确' if test_case['expected'] else '错误'}")
        
        # 执行检查
        is_correct, reason = evaluator.check_answer(
            test_case['question'],
            test_case['student_answer'],
            config.PROMPTS['answer_check']
        )
        
        print(f"\n实际判定: {'正确' if is_correct else '错误'}")
        print(f"判定理由: {reason[:200]}{'...' if len(reason) > 200 else ''}")
        
        # 验证结果
        if is_correct == test_case['expected']:
            print(f"\n✅ 测试通过")
            passed += 1
        else:
            print(f"\n❌ 测试失败（期望 {'正确' if test_case['expected'] else '错误'}，实际 {'正确' if is_correct else '错误'}）")
            failed += 1
    
    # 输出统计
    print("\n" + "="*60)
    print("测试统计")
    print("="*60)
    print(f"总测试数: {len(test_cases)}")
    print(f"通过: {passed}")
    print(f"失败: {failed}")
    print(f"通过率: {passed/len(test_cases)*100:.1f}%")
    
    if failed == 0:
        print("\n🎉 所有测试通过！")
    else:
        print(f"\n⚠️  有 {failed} 个测试失败，请检查")


if __name__ == "__main__":
    try:
        test_answer_checking()
    except KeyboardInterrupt:
        print("\n\n测试被中断")
    except Exception as e:
        logger.error(f"测试过程出错: {e}", exc_info=True)