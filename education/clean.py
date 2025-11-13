#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2025 AI系统与应用课题组@SRIBD

基于LLM和知识图谱协同的个性化出题系统 (PersonalExam)
Personalized Question Generation System Based on LLM and Knowledge Graph Collaboration

NPU显存清理脚本
在启动系统前运行此脚本清理显存
"""

import torch
import gc

try:
    import torch_npu
    
    print("🧹 正在清理NPU显存...")
    
    if torch.npu.is_available():
        npu_count = torch.npu.device_count()
        print(f"✅ 检测到 {npu_count} 个NPU设备")
        
        # 清理每个NPU的显存
        for i in range(npu_count):
            print(f"  清理 NPU {i}...")
            with torch.npu.device(f"npu:{i}"):
                torch.npu.empty_cache()
                torch.npu.synchronize()
        
        # Python垃圾回收
        gc.collect()
        
        print("✅ NPU显存清理完成！")
        
        # 显示当前显存状态
        print("\n📊 当前NPU显存状态:")
        for i in range(npu_count):
            props = torch.npu.get_device_properties(i)
            total_memory = props.total_memory / (1024**3)
            allocated = torch.npu.memory_allocated(i) / (1024**3)
            reserved = torch.npu.memory_reserved(i) / (1024**3)
            
            print(f"  NPU {i}:")
            print(f"    总容量: {total_memory:.2f} GB")
            print(f"    已分配: {allocated:.2f} GB")
            print(f"    已保留: {reserved:.2f} GB")
            print(f"    空闲: {total_memory - reserved:.2f} GB")
    else:
        print("❌ NPU不可用")
        
except ImportError:
    print("❌ torch_npu未安装")
except Exception as e:
    print(f"❌ 清理失败: {e}")
    import traceback
    traceback.print_exc()