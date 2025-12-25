# -*- coding: utf-8 -*-
"""
数据库管理模块
支持SQLite和MySQL
"""

import sqlite3
import json
import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self, db_path: str = "./data/education_system.db"):
        """
        初始化数据库
        
        Args:
            db_path: SQLite数据库路径
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = None
        
        self._init_database()
        logger.info(f"✅ 数据库初始化完成: {db_path}")
    
    @contextmanager
    def get_connection(self):
        """获取数据库连接（上下文管理器）"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"数据库操作失败: {e}")
            raise
        finally:
            conn.close()
    
    def _init_database(self):
        """初始化数据库表结构"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('student', 'teacher')),
                    real_name TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP
                )
            ''')
            
    
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS questions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question_no INTEGER UNIQUE,
                    content TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    explanation TEXT,
                    difficulty REAL DEFAULT 0.5,
                    major_point TEXT NOT NULL,
                    minor_point TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
      
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS student_states (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id TEXT NOT NULL,
                    major_point TEXT NOT NULL,
                    minor_point TEXT NOT NULL,
                    mastery_prob REAL DEFAULT 0.3,
                    p_init REAL DEFAULT 0.3,
                    p_learn REAL DEFAULT 0.2,
                    p_guess REAL DEFAULT 0.3,
                    p_slip REAL DEFAULT 0.1,
                    p_forget REAL DEFAULT 0.05,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(student_id, major_point, minor_point)
                )
            ''')
            
            # 4. 答题历史表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS answer_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id TEXT NOT NULL,
                    question_no INTEGER NOT NULL,
                    major_point TEXT NOT NULL,
                    minor_point TEXT NOT NULL,
                    student_answer TEXT,
                    is_correct BOOLEAN NOT NULL,
                    difficulty REAL,
                    mastery_before REAL,
                    mastery_after REAL,
                    answered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (question_no) REFERENCES questions(question_no)
                )
            ''')
            
            # 5. 最近表现记录表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS recent_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id TEXT NOT NULL,
                    major_point TEXT NOT NULL,
                    minor_point TEXT NOT NULL,
                    is_correct BOOLEAN NOT NULL,
                    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 创建索引
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_questions_points ON questions(major_point, minor_point)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_student_states ON student_states(student_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_answer_history ON answer_history(student_id, answered_at)')
            
            logger.info("✅ 数据库表结构创建完成")
    
    # ==================== 用户管理 ====================
    
    def _hash_password(self, password: str) -> str:
        """密码哈希"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def create_user(self, username: str, password: str, role: str, 
                    real_name: str = None) -> bool:
        """
        创建用户
        
        Args:
            username: 用户名
            password: 密码
            role: 角色 (student/teacher)
            real_name: 真实姓名
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                password_hash = self._hash_password(password)
                cursor.execute('''
                    INSERT INTO users (username, password_hash, role, real_name)
                    VALUES (?, ?, ?, ?)
                ''', (username, password_hash, role, real_name))
                logger.info(f"✅ 创建用户成功: {username} ({role})")
                return True
        except sqlite3.IntegrityError:
            logger.warning(f"⚠️  用户名已存在: {username}")
            return False
        except Exception as e:
            logger.error(f"❌ 创建用户失败: {e}")
            return False
    
    def verify_user(self, username: str, password: str) -> Optional[Dict]:
        """
        验证用户登录
        
        Returns:
            用户信息字典，验证失败返回None
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                password_hash = self._hash_password(password)
                cursor.execute('''
                    SELECT id, username, role, real_name, created_at
                    FROM users 
                    WHERE username = ? AND password_hash = ?
                ''', (username, password_hash))
                
                row = cursor.fetchone()
                if row:
                    # 更新最后登录时间
                    cursor.execute('''
                        UPDATE users SET last_login = CURRENT_TIMESTAMP
                        WHERE username = ?
                    ''', (username,))
                    
                    user_info = dict(row)
                    logger.info(f"✅ 用户登录成功: {username} ({user_info['role']})")
                    return user_info
                else:
                    logger.warning(f"⚠️  登录失败: {username}")
                    return None
        except Exception as e:
            logger.error(f"❌ 验证用户失败: {e}")
            return None
    
    def get_all_students(self) -> List[Dict]:
        """获取所有学生列表"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT username, real_name, created_at, last_login
                FROM users
                WHERE role = 'student'
                ORDER BY username
            ''')
            return [dict(row) for row in cursor.fetchall()]
    
    # ==================== 题库管理 ====================
    
# 在 database.py 的 insert_question 方法中修改字段获取逻辑

    def insert_question(self, question_data: Dict) -> bool:
        """插入题目 - 支持多种字段命名"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
            
            # 兼容多种字段命名
            # 题号
                question_no = question_data.get('题号')
            
            # 问题内容 (支持 "问题" 或 "题目")
                content = question_data.get('问题') or question_data.get('题目')
                if not content:
                    logger.warning(f"⚠️  题目{question_no}缺少问题内容,跳过")
                    return False
            
            # 答案
                answer = question_data.get('答案')
                if not answer:
                    logger.warning(f"⚠️  题目{question_no}缺少答案,跳过")
                    return False
            
            # 解析
                explanation = question_data.get('解析', '')
            
            # 难度
                difficulty = question_data.get('难度', 0.5)
            
            # 知识点 (支持多种命名)
                major_point = (question_data.get('知识点大类') or 
                              question_data.get('knowledge_point_major') or
                              question_data.get('知识点', '未分类'))
            
                minor_point = (question_data.get('知识点小类') or
                              question_data.get('knowledge_point_minor') or
                              question_data.get('知识点', '未分类'))
            
                cursor.execute('''
                    INSERT INTO questions 
                    (question_no, content, answer, explanation, difficulty, major_point, minor_point)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    question_no,
                    content,
                    answer,
                    explanation,
                    difficulty,
                    major_point,
                    minor_point
                ))
                return True
            
        except sqlite3.IntegrityError:
            logger.warning(f"⚠️  题目已存在: {question_data.get('题号')}")
            return False
        except Exception as e:
            logger.error(f"❌ 插入题目失败: {e}")
            logger.debug(f"问题数据: {question_data}")
            return False
    
    def get_questions_filtered(self, major_point: str = None, 
                               minor_point: str = None,
                               difficulty_range: Tuple[float, float] = None,
                               limit: int = None) -> List[Dict]:
        """获取筛选后的题目"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM questions WHERE 1=1"
            params = []
            
            if major_point:
                query += " AND major_point = ?"
                params.append(major_point)
            
            if minor_point:
                query += " AND minor_point = ?"
                params.append(minor_point)
            
            if difficulty_range:
                query += " AND difficulty >= ? AND difficulty < ?"
                params.extend(difficulty_range)
            
            if limit:
                query += f" LIMIT {limit}"
            
            cursor.execute(query, params)
            
            # 转换为原格式
            results = []
            for row in cursor.fetchall():
                results.append({
                    '题号': row['question_no'],
                    '问题': row['content'],
                    '答案': row['answer'],
                    '解析': row['explanation'],
                    '难度': row['difficulty'],
                    '知识点大类': row['major_point'],
                    '知识点小类': row['minor_point'],
                    'knowledge_point_major': row['major_point'],
                    'knowledge_point_minor': row['minor_point']
                })
            
            return results
    
    def get_all_questions(self) -> List[Dict]:
        """获取所有题目"""
        return self.get_questions_filtered()
    
    def get_knowledge_points(self) -> Dict[str, List[str]]:
        """获取所有知识点层级"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT DISTINCT major_point, minor_point
                FROM questions
                ORDER BY major_point, minor_point
            ''')
            
            knowledge_points = {}
            for row in cursor.fetchall():
                major = row['major_point']
                minor = row['minor_point']
                if major not in knowledge_points:
                    knowledge_points[major] = []
                if minor not in knowledge_points[major]:
                    knowledge_points[major].append(minor)
            
            return knowledge_points
    
    def get_question_statistics(self) -> Dict[str, Any]:
        """获取题库统计"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # 总题目数
            cursor.execute("SELECT COUNT(*) as total FROM questions")
            total = cursor.fetchone()['total']
            
            # 知识点大类分布
            cursor.execute('''
                SELECT major_point, COUNT(*) as count
                FROM questions
                GROUP BY major_point
            ''')
            major_dist = {row['major_point']: row['count'] for row in cursor.fetchall()}
            
            # 知识点小类分布
            cursor.execute('''
                SELECT minor_point, COUNT(*) as count
                FROM questions
                GROUP BY minor_point
            ''')
            minor_dist = {row['minor_point']: row['count'] for row in cursor.fetchall()}
            
            # 难度分布
            cursor.execute('''
                SELECT 
                    SUM(CASE WHEN difficulty < 0.35 THEN 1 ELSE 0 END) as easy,
                    SUM(CASE WHEN difficulty >= 0.35 AND difficulty < 0.65 THEN 1 ELSE 0 END) as medium,
                    SUM(CASE WHEN difficulty >= 0.65 THEN 1 ELSE 0 END) as hard
                FROM questions
            ''')
            diff_row = cursor.fetchone()
            
            return {
                '总题目数': total,
                '知识点大类分布': major_dist,
                '知识点小类分布': minor_dist,
                '难度分布': {
                    '简单': diff_row['easy'],
                    '中等': diff_row['medium'],
                    '困难': diff_row['hard']
                }
            }
    
    # ==================== 学生状态管理 ====================
    
    def get_student_state(self, student_id: str, major_point: str, 
                         minor_point: str) -> Optional[Dict]:
        """获取学生状态"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM student_states
                WHERE student_id = ? AND major_point = ? AND minor_point = ?
            ''', (student_id, major_point, minor_point))
            
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None
    
    def upsert_student_state(self, student_id: str, major_point: str,
                            minor_point: str, mastery_prob: float,
                            params: Dict = None) -> bool:
        """插入或更新学生状态"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                if params is None:
                    params = {}
                
                cursor.execute('''
                    INSERT INTO student_states 
                    (student_id, major_point, minor_point, mastery_prob, 
                     p_init, p_learn, p_guess, p_slip, p_forget, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(student_id, major_point, minor_point) 
                    DO UPDATE SET 
                        mastery_prob = ?,
                        updated_at = CURRENT_TIMESTAMP
                ''', (
                    student_id, major_point, minor_point, mastery_prob,
                    params.get('p_init', 0.3),
                    params.get('p_learn', 0.2),
                    params.get('p_guess', 0.3),
                    params.get('p_slip', 0.1),
                    params.get('p_forget', 0.05),
                    mastery_prob
                ))
                return True
        except Exception as e:
            logger.error(f"❌ 更新学生状态失败: {e}")
            return False
    
    def get_student_all_states(self, student_id: str) -> Dict[str, Dict[str, float]]:
        """获取学生所有知识点状态"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT major_point, minor_point, mastery_prob
                FROM student_states
                WHERE student_id = ?
            ''', (student_id,))
            
            states = {}
            for row in cursor.fetchall():
                major = row['major_point']
                minor = row['minor_point']
                if major not in states:
                    states[major] = {}
                states[major][minor] = row['mastery_prob']
            
            return states
    
    def get_weak_points(self, student_id: str, 
                       threshold: float = 0.4) -> List[Tuple[str, str, float]]:
        """获取薄弱知识点"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT major_point, minor_point, mastery_prob
                FROM student_states
                WHERE student_id = ? AND mastery_prob < ?
                ORDER BY mastery_prob ASC
            ''', (student_id, threshold))
            
            return [(row['major_point'], row['minor_point'], row['mastery_prob']) 
                    for row in cursor.fetchall()]
    
    # ==================== 答题历史管理 ====================
    
    def insert_answer_record(self, student_id: str, question_no: int,
                            major_point: str, minor_point: str,
                            student_answer: str, is_correct: bool,
                            difficulty: float, mastery_before: float,
                            mastery_after: float) -> bool:
        """插入答题记录"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO answer_history
                    (student_id, question_no, major_point, minor_point,
                     student_answer, is_correct, difficulty, mastery_before, mastery_after)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (student_id, question_no, major_point, minor_point,
                      student_answer, is_correct, difficulty, mastery_before, mastery_after))
                
                # 同时更新recent_performance
                cursor.execute('''
                    INSERT INTO recent_performance
                    (student_id, major_point, minor_point, is_correct)
                    VALUES (?, ?, ?, ?)
                ''', (student_id, major_point, minor_point, is_correct))
                
                # 只保留最近10条记录
                cursor.execute('''
                    DELETE FROM recent_performance
                    WHERE student_id = ? AND major_point = ? AND minor_point = ?
                    AND id NOT IN (
                        SELECT id FROM recent_performance
                        WHERE student_id = ? AND major_point = ? AND minor_point = ?
                        ORDER BY recorded_at DESC
                        LIMIT 10
                    )
                ''', (student_id, major_point, minor_point, 
                      student_id, major_point, minor_point))
                
                return True
        except Exception as e:
            logger.error(f"❌ 插入答题记录失败: {e}")
            return False
    
    def get_answer_history(self, student_id: str, major_point: str = None,
                          minor_point: str = None, limit: int = None) -> List[Dict]:
        """获取答题历史"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            query = '''
                SELECT ah.*, q.content as question_content, q.answer as correct_answer
                FROM answer_history ah
                LEFT JOIN questions q ON ah.question_no = q.question_no
                WHERE ah.student_id = ?
            '''
            params = [student_id]
            
            if major_point:
                query += " AND ah.major_point = ?"
                params.append(major_point)
            
            if minor_point:
                query += " AND ah.minor_point = ?"
                params.append(minor_point)
            
            query += " ORDER BY ah.answered_at DESC"
            
            if limit:
                query += f" LIMIT {limit}"
            
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    def get_recent_performance(self, student_id: str, major_point: str,
                              minor_point: str) -> List[bool]:
        """获取最近表现"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT is_correct
                FROM recent_performance
                WHERE student_id = ? AND major_point = ? AND minor_point = ?
                ORDER BY recorded_at DESC
                LIMIT 10
            ''', (student_id, major_point, minor_point))
            
            return [row['is_correct'] for row in cursor.fetchall()]
    
    def get_student_profile(self, student_id: str) -> Dict[str, Any]:
        """生成学生档案"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # 获取所有状态
            states = self.get_student_all_states(student_id)
            
            if not states:
                return {
                    'student_id': student_id,
                    'knowledge_points': {},
                    'overall_mastery': 0.0,
                    'weak_points': [],
                    'strengths': [],
                    'total_knowledge_points': 0,
                    'total_answers': 0
                }
            
            # 计算整体掌握度
            all_mastery = []
            for major_dict in states.values():
                all_mastery.extend(major_dict.values())
            
            overall_mastery = sum(all_mastery) / len(all_mastery) if all_mastery else 0.0
            
            # 获取薄弱点和强项
            weak_points = self.get_weak_points(student_id, 0.4)
            
            cursor.execute('''
                SELECT major_point, minor_point, mastery_prob
                FROM student_states
                WHERE student_id = ? AND mastery_prob > 0.7
            ''', (student_id,))
            strengths = [(row['major_point'], row['minor_point'], row['mastery_prob'])
                        for row in cursor.fetchall()]
            
            # 获取答题总数
            cursor.execute('''
                SELECT COUNT(*) as total
                FROM answer_history
                WHERE student_id = ?
            ''', (student_id,))
            total_answers = cursor.fetchone()['total']
            
            return {
                'student_id': student_id,
                'knowledge_points': states,
                'overall_mastery': overall_mastery,
                'weak_points': weak_points[:5],
                'strengths': strengths,
                'total_knowledge_points': len(all_mastery),
                'total_answers': total_answers
            }
    
    # ==================== 数据迁移 ====================
    
    def migrate_from_json(self, questions_file: str, states_file: str) -> Dict[str, int]:
        """
        从JSON文件迁移数据
        
        Returns:
            迁移统计信息
        """
        stats = {'questions': 0, 'states': 0, 'history': 0}
        
        # 1. 迁移题库
        logger.info("🔄 迁移题库数据...")
        try:
            with open(questions_file, 'r', encoding='utf-8') as f:
                questions = json.load(f)
            
            for q in questions:
                if self.insert_question(q):
                    stats['questions'] += 1
            
            logger.info(f"✅ 题库迁移完成: {stats['questions']} 道题")
        except Exception as e:
            logger.error(f"❌ 题库迁移失败: {e}")
        
        # 2. 迁移学生状态
        logger.info("🔄 迁移学生状态数据...")
        try:
            with open(states_file, 'r', encoding='utf-8') as f:
                states_data = json.load(f)
            
            for student_id, major_dict in states_data.items():
                for major_point, minor_dict in major_dict.items():
                    for minor_point, state in minor_dict.items():
                        # 插入状态
                        params = state.get('params', {})
                        self.upsert_student_state(
                            student_id, major_point, minor_point,
                            state['mastery_prob'], params
                        )
                        stats['states'] += 1
                        
                        # 插入答题历史
                        for record in state.get('answer_history', []):
                            q = record.get('question', {})
                            self.insert_answer_record(
                                student_id,
                                q.get('题号', 0),
                                major_point,
                                minor_point,
                                '',
                                record.get('is_correct', False),
                                record.get('difficulty', 0.5),
                                record.get('mastery_before', 0.3),
                                record.get('mastery_after', 0.3)
                            )
                            stats['history'] += 1
            
            logger.info(f"✅ 学生状态迁移完成: {stats['states']} 条记录, {stats['history']} 条历史")
        except Exception as e:
            logger.error(f"❌ 学生状态迁移失败: {e}")
        
        return stats


def create_database_manager(db_path: str = "./data/education_system.db") -> DatabaseManager:
    """创建数据库管理器"""
    return DatabaseManager(db_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 测试数据库
    db = create_database_manager()
    
    # 创建测试用户
    db.create_user("student001", "123456", "student", "张三")
    db.create_user("teacher001", "123456", "teacher", "李老师")
    
    # 测试登录
    user = db.verify_user("student001", "123456")
    print(f"登录测试: {user}")
    
    # 测试迁移
    # stats = db.migrate_from_json(
    #     "./data/question_database_2.json",
    #     "./data/student_states.json"
    # )
    # print(f"迁移统计: {stats}")