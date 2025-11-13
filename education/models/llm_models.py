# -*- coding: utf-8 -*-
"""
Copyright (c) 2025 AI系统及应用课题组@SRIBD

基于LLM和知识图谱协同的个性化出题系统 (PersonalExam)
Personalized Question Generation System Based on LLM and Knowledge Graph Collaboration

LLM模型接口模块
统一使用PanGu-7B模型，支持昇腾NPU加速和多NPU并行
优化：单例模式避免重复加载，支持多NPU负载均衡
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Dict, Any
import logging
import re
import threading

logger = logging.getLogger(__name__)

# 全局模型实例缓存(单例模式)
_MODEL_INSTANCE_CACHE = {}
_CACHE_LOCK = threading.Lock()


class PanGuModel:
    """盘古7B模型封装 - 统一用于出题和评估"""
    
    def __init__(self, model_path: str, config: Dict[str, Any]):
        """
        初始化盘古7B模型
        
        Args:
            model_path: 模型路径
            config: 模型配置字典
        """
        self.model_path = model_path
        self.config = config
        
        # 检查NPU可用性并配置多NPU
        self.devices = self._setup_devices(config.get("device", "npu"))
        self.current_device_idx = 0  # 用于轮询负载均衡
        
        logger.info(f"盘古7B模型设备配置: {self.devices}")
        
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        
        # 盘古7B特殊配置
        self.system_prompt = config.get("system_prompt", "")
        self.eos_token_id = config.get("eos_token_id", 45892)
        self.enable_thinking = config.get("enable_thinking", False)
        
    def _setup_devices(self, device_config: str) -> list:
        """设置计算设备，支持多NPU"""
        if device_config == "npu":
            try:
                import torch_npu
                if torch.npu.is_available():
                    npu_count = torch.npu.device_count()
                    logger.info(f"✅ 检测到 {npu_count} 个NPU设备")
                    
                    # 返回所有可用NPU设备列表
                    devices = [f"npu:{i}" for i in range(npu_count)]
                    
                    # 显示每个NPU的显存信息
                    for i in range(npu_count):
                        props = torch.npu.get_device_properties(i)
                        total_memory = props.total_memory / (1024**3)  # 转换为GB
                        logger.info(f"  NPU {i}: {props.name}, 总显存: {total_memory:.2f} GB")
                    
                    return devices
                else:
                    logger.error("❌ NPU不可用")
                    raise RuntimeError("NPU不可用，系统终止")
            except ImportError:
                logger.error("❌ torch_npu未安装")
                raise RuntimeError("torch_npu未安装，系统终止")
        elif device_config == "cuda" and torch.cuda.is_available():
            return ["cuda:0"]
        else:
            logger.error("❌ 不支持的设备配置或设备不可用")
            raise RuntimeError("设备不可用，系统终止")
    
    def _get_next_device(self) -> str:
        if len(self.devices) == 1:
            return self.devices[0]
        
        # 轮询方式选择设备
        device = self.devices[self.current_device_idx]
        self.current_device_idx = (self.current_device_idx + 1) % len(self.devices)
        return device
            
    def load_model(self):
        """加载模型和tokenizer - 只加载一次"""
        if self.is_loaded:
            logger.info("✅ 模型已加载，跳过重复加载")
            return
        
        try:
            logger.info(f"🔄 正在加载盘古7B模型: {self.model_path}")
            
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                use_fast=False,
                trust_remote_code=True,
                local_files_only=False
            )
            
            # 设置数据类型
            torch_dtype = torch.float16
            
            # 多NPU策略
            if len(self.devices) > 1:
                logger.info(f"🚀 使用多NPU模式，跨{len(self.devices)}个设备分布模型")
                
                try:
                    # 方案1: 使用accelerate的自动设备映射
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        trust_remote_code=True,
                        torch_dtype=torch_dtype,
                        device_map="auto",  # 自动分配到多个NPU
                        max_memory={i: "50GB" for i in range(len(self.devices))},  # 每个NPU最大使用50GB
                        local_files_only=False,
                        low_cpu_mem_usage=True  # 减少CPU内存使用
                    )
                    logger.info("✅ 使用device_map='auto'模式成功加载")
                except Exception as e:
                    logger.warning(f"⚠️  device_map='auto'加载失败: {e}")
                    logger.info("📍 回退到单NPU模式")
                    
                    # 回退到单NPU模式
                    import torch_npu
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        trust_remote_code=True,
                        torch_dtype=torch_dtype,
                        local_files_only=False
                    )
                    self.model = self.model.to(self.devices[0])
                    self.devices = [self.devices[0]]  # 更新为单设备
            else:
                # 单NPU模式
                logger.info(f"📍 使用单NPU模式: {self.devices[0]}")
                import torch_npu
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                    torch_dtype=torch_dtype,
                    local_files_only=False
                )
                self.model = self.model.to(self.devices[0])
            
            self.model.eval()
            self.is_loaded = True
            
            # 显示模型分布情况
            if hasattr(self.model, 'hf_device_map'):
                logger.info(f"📊 模型分布情况:")
                device_summary = {}
                for layer, device in self.model.hf_device_map.items():
                    device_summary[device] = device_summary.get(device, 0) + 1
                for device, count in sorted(device_summary.items()):
                    logger.info(f"   {device}: {count} 层")
            else:
                logger.info(f"📊 模型位于: {self.devices[0]}")
            
            logger.info(f"✅ 盘古7B模型加载成功")
            
        except Exception as e:
            logger.error(f"❌ 盘古7B模型加载失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise RuntimeError(f"模型加载失败: {e}")
    
    def _build_chat_prompt(self, user_prompt: str, enable_thinking: bool = None) -> str:
        """
        构建符合盘古7B格式的对话提示词
        
        Args:
            user_prompt: 用户输入
            enable_thinking: 是否启用思维链
            
        Returns:
            格式化的提示词
        """
        if enable_thinking is None:
            enable_thinking = self.enable_thinking
        
        # 根据是否启用思维链添加特殊标记
        thinking_flag = " /auto_think" if enable_thinking else " /no_think"
        
        # 构建消息列表
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt + thinking_flag}
        ]
        
        # 使用tokenizer的chat template
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            return prompt
        except Exception as e:
            logger.warning(f"应用chat template失败: {e}，使用简单格式")
            return f"{self.system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"
    
    def _parse_pangu_output(self, output_text: str) -> str:

        try:
            # 盘古7B使用特殊token分隔思维和输出
            if "[unused17]" in output_text:
                # 提取思维内容（如果需要）
                if "[unused16]" in output_text:
                    thinking_content = output_text.split("[unused17]")[0].split("[unused16]")[-1].strip()
                    if thinking_content:
                        logger.debug(f"思维过程: {thinking_content[:100]}...")
                
                # 提取实际输出
                content = output_text.split("[unused17]")[-1].split("[unused10]")[0].strip()
                return content
            else:
                # 没有特殊token，直接返回清理后的文本
                content = output_text.replace("[unused16]", "").replace("[unused17]", "").replace("[unused10]", "").strip()
                return content
                
        except Exception as e:
            logger.warning(f"解析盘古输出失败: {e}，返回原始文本")
            return output_text
            
    def generate(self, prompt: str, 
                max_length: Optional[int] = None,
                temperature: Optional[float] = None,
                top_p: Optional[float] = None,
                enable_thinking: Optional[bool] = None) -> str:
        """
        生成文本
        
        Args:
            prompt: 输入提示
            max_length: 最大生成长度
            temperature: 温度参数
            top_p: top_p采样参数
            enable_thinking: 是否启用思维链
            
        Returns:
            生成的文本
        """
        # 确保模型已加载
        if not self.is_loaded:
            self.load_model()
            
        max_new_tokens = max_length or self.config.get("max_new_tokens", 4096)
        temperature = temperature or self.config.get("temperature", 0.7)
        top_p = top_p or self.config.get("top_p", 0.9)
        
        try:
            # 构建提示词
            formatted_prompt = self._build_chat_prompt(prompt, enable_thinking)
            
            # Tokenize
            model_inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
            
            # 根据模型加载方式决定如何处理输入
            if hasattr(self.model, 'hf_device_map'):
                # 多NPU模式（使用device_map="auto"）
                # 将输入移动到第一个模型层所在的设备
                first_device = list(self.model.hf_device_map.values())[0]
                model_inputs = {k: v.to(first_device) for k, v in model_inputs.items()}
            else:
                # 单NPU模式
                import torch_npu
                model_inputs = {k: v.to(self.devices[0]) for k, v in model_inputs.items()}
            
            # 生成（优化：低温度时关闭采样以提升速度）
            with torch.no_grad():
                # 当temperature很低时，使用贪心解码（更快）
                do_sample = temperature > 0.1
                outputs = self.model.generate(
                    **model_inputs,
                    max_new_tokens=min(max_new_tokens, 512),  # 限制最大长度以提升速度
                    temperature=temperature if do_sample else None,
                    top_p=top_p if do_sample else None,
                    do_sample=do_sample,
                    eos_token_id=self.eos_token_id,
                    pad_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=True
                )
            
            # 解码
            input_length = model_inputs['input_ids'].shape[1]
            generated_tokens = outputs.sequences[:, input_length:]
            
            # 移回CPU进行解码
            generated_tokens = generated_tokens.cpu()
            
            output_text = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=False)
            
            # 解析输出
            response = self._parse_pangu_output(output_text)
            
            logger.debug(f"✅ 生成完成，输出长度: {len(response)}")
            return response
            
        except Exception as e:
            logger.error(f"❌ 文本生成失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise RuntimeError(f"生成失败: {str(e)}")
    
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            del self.tokenizer
        
        # 清理NPU缓存
        try:
            import torch_npu
            if torch.npu.is_available():
                for device in self.devices:
                    torch.npu.empty_cache()
        except:
            pass


def create_llm_model(model_type: str, model_path: str, config: Dict[str, Any]) -> PanGuModel:
    import os
    
    # 使用单例模式，避免重复加载
    cache_key = f"{model_type}_{model_path}"
    
    with _CACHE_LOCK:
        if cache_key in _MODEL_INSTANCE_CACHE:
            logger.info(f"♻️  返回已缓存的模型实例: {model_type}")
            return _MODEL_INSTANCE_CACHE[cache_key]
        
        # 统一使用盘古7B模型
        logger.info(f"🆕 创建新的盘古7B模型实例（任务类型: {model_type}）")
        
        # 检查模型路径
        if not os.path.exists(model_path):
            logger.error(f"❌ 模型路径不存在: {model_path}")
            raise RuntimeError(f"模型路径不存在: {model_path}")
        
        model = PanGuModel(model_path, config)
        
        # 缓存实例
        _MODEL_INSTANCE_CACHE[cache_key] = model
        
        return model


def clear_model_cache():
    """清理模型缓存（用于重新加载模型）"""
    global _MODEL_INSTANCE_CACHE
    with _CACHE_LOCK:
        for model in _MODEL_INSTANCE_CACHE.values():
            del model
        _MODEL_INSTANCE_CACHE.clear()
        logger.info("🗑️  模型缓存已清理")


if __name__ == "__main__":
    # 测试代码
    import sys
    sys.path.append("..")
    from config import PANGU_MODEL_PATH, PANGU_MODEL_CONFIG
    
    logging.basicConfig(level=logging.INFO)
    
    # 测试盘古模型
    pangu = create_llm_model('pangu', PANGU_MODEL_PATH, PANGU_MODEL_CONFIG)
    pangu.load_model()
    
    test_prompt = "请简要介绍一下贝叶斯定理"
    response = pangu.generate(test_prompt)
    print(f"响应: {response}")