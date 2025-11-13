# -*- coding: utf-8 -*-
"""
Copyright (c) 2025 AI系统与应用课题组@SRIBD

基于LLM和知识图谱协同的个性化出题系统 (PersonalExam)
Personalized Question Generation System Based on LLM and Knowledge Graph Collaboration

调试测试脚本 - 专门测试实体提取和选题逻辑
"""

import sys
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.DEBUG,  # 使用DEBUG级别
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_entity_extraction():
    """专门测试实体提取"""
    logger.info("=" * 70)
    logger.info("调试测试: 实体提取")
    logger.info("=" * 70)
    
    try:
        from config import BGE_M3_MODEL_PATH, PANGU_MODEL_PATH, EMBEDDING_MODEL_CONFIG, PANGU_MODEL_CONFIG
        from models.embedding_model import create_embedding_model
        from models.llm_models import create_llm_model
        from knowledge_management.rag_engine import LocalRAGEngine
        
        # 加载模型
        logger.info("加载模型...")
        embedding_model = create_embedding_model(BGE_M3_MODEL_PATH, EMBEDDING_MODEL_CONFIG)
        embedding_model.load_model()
        
        llm_model = create_llm_model('pangu', PANGU_MODEL_PATH, PANGU_MODEL_CONFIG)
        llm_model.load_model()
        
        # 创建RAG引擎
        rag = LocalRAGEngine(embedding_model, llm_model)
        
        # 测试文本
        test_context = """
题目1: 一元二次方程
问题: 解方程 x^2 - 5x + 6 = 0
答案: x = 2 或 x = 3
解析: 使用因式分解法，将方程改写为 (x-2)(x-3) = 0

题目2: 因式分解
问题: 分解因式 x^2 - 4
答案: (x+2)(x-2)
解析: 使用平方差公式
"""
        
        # 测试提取
        logger.info("\n开始测试实体提取...")
        logger.info(f"输入文本长度: {len(test_context)} 字符")
        
        result = rag.extract_entities_and_relations(test_context)
        
        logger.info("\n" + "=" * 70)
        logger.info("提取结果:")
        logger.info("=" * 70)
        
        logger.info(f"\n实体数量: {len(result['entities'])}")
        for i, entity in enumerate(result['entities'], 1):
            logger.info(f"  {i}. {entity.get('name')} ({entity.get('type')})")
        
        logger.info(f"\n关系数量: {len(result['relations'])}")
        for i, relation in enumerate(result['relations'], 1):
            logger.info(f"  {i}. {relation.get('source')} -> {relation.get('target')} ({relation.get('relation')})")
        
        return True
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_question_selection_detailed():
    """详细测试题目选择"""
    logger.info("\n" + "=" * 70)
    logger.info("调试测试: 题目选择详细流程")
    logger.info("=" * 70)
    
    try:
        from config import (BGE_M3_MODEL_PATH, PANGU_MODEL_PATH, 
                           EMBEDDING_MODEL_CONFIG, PANGU_MODEL_CONFIG, QUESTION_DB)
        from models.embedding_model import create_embedding_model
        from models.llm_models import create_llm_model
        from data_management.question_db import create_question_database
        from knowledge_management.rag_engine import create_rag_engine
        from utils.question_generator import create_question_selector
        
        # 创建组件
        logger.info("创建组件...")
        embedding_model = create_embedding_model(BGE_M3_MODEL_PATH, EMBEDDING_MODEL_CONFIG)
        embedding_model.load_model()
        
        llm_model = create_llm_model('pangu', PANGU_MODEL_PATH, PANGU_MODEL_CONFIG)
        llm_model.load_model()
        
        question_db = create_question_database(str(QUESTION_DB))
        rag_engine = create_rag_engine(embedding_model, llm_model)
        
        # 构建索引
        logger.info("构建索引...")
        all_questions = question_db.get_all_questions()
        rag_engine.build_question_index(all_questions)
        logger.info(f"索引完成: {len(all_questions)} 道题")
        
        # 创建选择器
        selector = create_question_selector(rag_engine, llm_model, question_db)
        
        # 测试不同掌握度
        test_cases = [
            ("基础薄弱", 0.2, "代数", "一元二次方程"),
            ("中等水平", 0.5, "代数", "一元二次方程"),
            ("掌握良好", 0.8, "代数", "一元二次方程"),
        ]
        
        for case_name, mastery, major, minor in test_cases:
            logger.info("\n" + "-" * 70)
            logger.info(f"测试场景: {case_name} (掌握度: {mastery:.1%})")
            logger.info("-" * 70)
            
            # 1. 先看看RAG检索结果
            logger.info("\n步骤1: RAG检索")
            query = f"{major} {minor} {'简单' if mastery < 0.3 else '中等' if mastery < 0.7 else '困难'}"
            logger.info(f"  查询: {query}")
            
            search_results = rag_engine.search_questions(
                query=query,
                major_point=major,
                minor_point=minor,
                top_k=3
            )
            logger.info(f"  检索到 {len(search_results)} 道题:")
            for i, item in enumerate(search_results, 1):
                q = item['question']
                logger.info(f"    {i}. 题号{q.get('题号')} (难度:{q.get('难度'):.2f}, 相似度:{item['score']:.3f})")
            
            # 2. 知识子图
            logger.info("\n步骤2: 构建知识子图")
            subgraph = rag_engine.build_knowledge_subgraph(mastery, major, minor, top_k=3)
            logger.info(f"  实体数: {len(subgraph['entities'])}")
            logger.info(f"  实体: {[e['name'] for e in subgraph['entities'][:5]]}")
            
            # 3. 智能选题
            logger.info("\n步骤3: 智能选题")
            selected = selector.select_question(
                student_id="debug_test",
                student_mastery=mastery,
                major_point=major,
                minor_point=minor,
                used_question_ids=set(),
                top_k=3
            )
            
            if selected:
                logger.info(f"  ✅ 选中: 题号{selected.get('题号')} (难度:{selected.get('难度'):.2f})")
                logger.info(f"  问题: {selected.get('问题')[:80]}...")
            else:
                logger.error("  ❌ 选题失败")
        
        return True
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_pangu_response_parsing():
    """测试盘古7B响应解析"""
    logger.info("\n" + "=" * 70)
    logger.info("调试测试: 盘古7B响应解析")
    logger.info("=" * 70)
    
    # 模拟各种可能的响应格式
    test_responses = [
        # 标准JSON
        '''{"entities": [{"name": "一元二次方程", "type": "知识点"}], "relations": []}''',
        
        # 带前后文字
        '''好的，我来提取：
        {"entities": [{"name": "因式分解", "type": "方法"}], "relations": []}
        以上是提取结果。''',
        
        # 换行格式
        '''{
            "entities": [
                {"name": "配方法", "type": "方法"}
            ],
            "relations": []
        }''',
        
        # 不标准的格式
        '''实体：一元二次方程、因式分解、求根公式
        关系：一元二次方程可以使用因式分解''',
    ]
    
    from knowledge_management.rag_engine import LocalRAGEngine
    
    # 创建临时实例测试解析
    class MockLLM:
        is_loaded = True
    
    rag = LocalRAGEngine(None, MockLLM())
    
    for i, response in enumerate(test_responses, 1):
        logger.info(f"\n测试响应 {i}:")
        logger.info(f"原文: {response[:100]}...")
        
        result = rag._parse_kg_response(response)
        logger.info(f"解析结果: {len(result['entities'])} 个实体, {len(result['relations'])} 个关系")
        if result['entities']:
            logger.info(f"  实体: {[e['name'] for e in result['entities']]}")
    
    return True


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("RAG系统调试测试")
    print("=" * 70)
    
    tests = [
        ("盘古响应解析", test_pangu_response_parsing),
        ("实体提取", test_entity_extraction),
        ("详细选题流程", test_question_selection_detailed),
    ]
    
    results = {}
    for test_name, test_func in tests:
        logger.info(f"\n{'='*70}")
        logger.info(f"运行测试: {test_name}")
        logger.info(f"{'='*70}")
        
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"测试崩溃: {e}")
            results[test_name] = False
    
    # 结果汇总
    print("\n" + "=" * 70)
    print("测试结果")
    print("=" * 70)
    
    for test_name, passed in results.items():
        status = "✅" if passed else "❌"
        print(f"{status} {test_name}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n✅ 所有测试通过")
    else:
        print("\n⚠️ 部分测试失败，请检查日志")


if __name__ == "__main__":
    main()