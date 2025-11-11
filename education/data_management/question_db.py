# -*- coding: utf-8 -*-
"""
Copyright (c) 2025 AI系统及应用课题组@SRIBD

基于LLM和知识图谱协同的个性化出题系统 (PersonalExam)
Personalized Question Generation System Based on LLM and Knowledge Graph Collaboration

"""

import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class EnhancedQuestionDatabase:
    
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.questions = []
        self.load_database()
        
        logger.info(f"✅ 增强版题库初始化完成: {db_path}")
    
    def load_database(self):
        """加载数据库"""
        if self.db_path.exists():
            try:
                with open(self.db_path, 'r', encoding='utf-8') as f:
                    self.questions = json.load(f)
                
                # 数据预处理：统一字段名
                for q in self.questions:
                    # 确保有知识点大类和小类
                    if 'knowledge_point_major' not in q and '知识点大类' in q:
                        q['knowledge_point_major'] = q['知识点大类']
                    if 'knowledge_point_minor' not in q and '知识点小类' in q:
                        q['knowledge_point_minor'] = q['知识点小类']
                    
                    # 处理难度字段（支持数值型）
                    if isinstance(q.get('难度'), (int, float)):
                        # 数值难度，保持不变
                        pass
                    
                logger.info(f"✅ 加载题库成功: {len(self.questions)} 道题目")
            except Exception as e:
                logger.error(f"❌ 加载题库失败: {e}")
                self.questions = []
        else:
            logger.warning(f"⚠️  题库文件不存在: {self.db_path}")
            self.questions = []
    
    def save_database(self):
        """保存数据库"""
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.db_path, 'w', encoding='utf-8') as f:
                json.dump(self.questions, f, ensure_ascii=False, indent=2)
            logger.info(f"💾 题库已保存: {len(self.questions)} 道题目")
        except Exception as e:
            logger.error(f"❌ 保存题库失败: {e}")
            raise
    
    def get_questions_by_minor_point(self, major_point: str, 
                                     minor_point: str) -> List[Dict[str, Any]]:
        """根据知识点小类获取题目"""
        results = []
        for q in self.questions:
            q_major = q.get('knowledge_point_major', q.get('知识点大类', ''))
            q_minor = q.get('knowledge_point_minor', q.get('知识点小类', ''))
            
            if q_major == major_point and q_minor == minor_point:
                results.append(q)
        
        return results
    
    def get_questions_by_major_point(self, major_point: str) -> List[Dict[str, Any]]:
        """根据知识点大类获取题目"""
        results = []
        for q in self.questions:
            q_major = q.get('knowledge_point_major', q.get('知识点大类', ''))
            if q_major == major_point:
                results.append(q)
        return results
    
    def get_questions_filtered(self, 
                               major_point: Optional[str] = None,
                               minor_point: Optional[str] = None,
                               difficulty_range: Optional[tuple] = None,
                               limit: Optional[int] = None) -> List[Dict[str, Any]]:

        filtered = self.questions.copy()
        
        if major_point:
            filtered = [q for q in filtered 
                       if q.get('knowledge_point_major', q.get('知识点大类', '')) == major_point]
        
        if minor_point:
            filtered = [q for q in filtered
                       if q.get('knowledge_point_minor', q.get('知识点小类', '')) == minor_point]
        
        if difficulty_range:
            min_diff, max_diff = difficulty_range
            filtered = [q for q in filtered
                       if min_diff <= q.get('难度', 0.5) < max_diff]
        
        if limit:
            filtered = filtered[:limit]
        
        logger.debug(f"筛选结果: {len(filtered)} 道题目")
        return filtered
    
    def get_all_knowledge_points(self) -> Dict[str, List[str]]:
        """获取所有知识点（大类 -> 小类列表）"""
        knowledge_points = {}
        
        for q in self.questions:
            major = q.get('knowledge_point_major', q.get('知识点大类', ''))
            minor = q.get('knowledge_point_minor', q.get('知识点小类', ''))
            
            if major:
                if major not in knowledge_points:
                    knowledge_points[major] = []
                if minor and minor not in knowledge_points[major]:
                    knowledge_points[major].append(minor)
        
        return knowledge_points
    
    def get_all_questions(self) -> List[Dict[str, Any]]:
        """获取所有题目"""
        return self.questions.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取题库统计"""
        stats = {
            "总题目数": len(self.questions),
            "知识点大类分布": {},
            "知识点小类分布": {},
            "难度分布": {"简单": 0, "中等": 0, "困难": 0}
        }
        
        for q in self.questions:
            # 大类统计
            major = q.get('knowledge_point_major', q.get('知识点大类', '未分类'))
            stats["知识点大类分布"][major] = stats["知识点大类分布"].get(major, 0) + 1
            
            # 小类统计
            minor = q.get('knowledge_point_minor', q.get('知识点小类', '未分类'))
            stats["知识点小类分布"][minor] = stats["知识点小类分布"].get(minor, 0) + 1
            
            # 难度统计
            diff = q.get('难度', 0.5)
            if isinstance(diff, (int, float)):
                if diff < 0.35:
                    stats["难度分布"]["简单"] += 1
                elif diff < 0.65:
                    stats["难度分布"]["中等"] += 1
                else:
                    stats["难度分布"]["困难"] += 1
        
        return stats
    
    def insert_question(self, question: Dict[str, Any]) -> bool:
        """插入题目"""
        try:
            if '题号' not in question:
                question['题号'] = len(self.questions) + 1
            question['创建时间'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.questions.append(question)
            self.save_database()
            logger.info(f"✅ 插入题目成功: 题号 {question['题号']}")
            return True
        except Exception as e:
            logger.error(f"❌ 插入题目失败: {e}")
            return False
    
    def import_from_json(self, json_path: str) -> int:
        """从JSON文件导入题目"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                new_questions = json.load(f)
            
            if not isinstance(new_questions, list):
                logger.error("❌ JSON格式错误,应为题目数组")
                return 0
            
            success_count = 0
            for q in new_questions:
                if self.insert_question(q):
                    success_count += 1
            
            logger.info(f"✅ 批量导入完成: {success_count}/{len(new_questions)} 道题目")
            return success_count
            
        except Exception as e:
            logger.error(f"❌ 导入题目失败: {e}")
            return 0


def create_question_database(db_path: str) -> EnhancedQuestionDatabase:
    """创建题库数据库实例"""
    return EnhancedQuestionDatabase(db_path)


if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from config import QUESTION_DB
    
    logging.basicConfig(level=logging.INFO)
    
    db = create_question_database(str(QUESTION_DB))
    
    stats = db.get_statistics()
    print(f"题库统计: {json.dumps(stats, ensure_ascii=False, indent=2)}")
    
    knowledge_points = db.get_all_knowledge_points()
    print(f"\n知识点层级:")
    for major, minors in knowledge_points.items():
        print(f"  {major}: {len(minors)} 个小类")