import torch
from transformers import GPT2Tokenizer, GPT2Model

class MyCustomCompressor:
    def __init__(self):
        # 1. 确保设备与推理机一致，优先使用 GPU 以加速矩阵分解
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 2. 加载基础组件
        # 使用 GPT2Model 专门用于提取隐空间嵌入 (Embedding Layer)
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2Model.from_pretrained("gpt2").to(self.device)
        self.model.eval()

    def compress(self, segments, mask_ratio):
        """
        核心算法：本征语义提取算法 (Intrinsic Semantic Extraction)
        包含：ZCA白化、零空间投影、SVD特征压缩与逆映射
        """
        # 1. 将碎片列表合并为完整字符串并转为 Token IDs
        full_text = "".join(segments)
        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            # 获取原始稠密向量 outputs 形状为 [1, Seq_len, 768]
            raw_embeddings = self.model.get_input_embeddings()(inputs.input_ids)
            
            # 降维为 2D 矩阵以进行线性代数运算: [Seq_len, 768]
            E = raw_embeddings.squeeze(0)
            seq_len, dim = E.shape
            
            # =========================================================
            # 阶段 1: 语义空间各向异性修正 (ZCA Whitening)
            # =========================================================
            # 1.1 均值中心化
            mean = E.mean(dim=0, keepdim=True)
            E_centered = E - mean
            
            # 1.2 计算协方差矩阵
            cov = torch.mm(E_centered.t(), E_centered) / (seq_len - 1 + 1e-5)
            
            # 1.3 特征分解 (使用 eigh 处理对称矩阵，提升数值稳定性)
            L, V = torch.linalg.eigh(cov)
            L = torch.clamp(L, min=1e-5) # 截断极小值，防止除零溢出
            
            # 1.4 计算白化矩阵 W 和 逆白化矩阵 W_inv
            # 公式: W = V * L^(-1/2) * V^T
            W = torch.mm(V, torch.mm(torch.diag(1.0 / torch.sqrt(L)), V.t()))
            W_inv = torch.mm(V, torch.mm(torch.diag(torch.sqrt(L)), V.t()))
            
            # 1.5 执行白化 (隐空间球化)
            E_white = torch.mm(E_centered, W)
            
            # =========================================================
            # 阶段 2: 零空间投影 (Null-space Projection)
            # =========================================================
            # 工程化妥协：由于无法在单次 compress 中动态加载大规模外部语料，
            # 这里通过对当前文本白化矩阵提取 Top-1 奇异向量，
            # 作为局部的“高频语言背景噪声”（通常是标点或高频停用词的共性模式）。
            U_lang, S_lang, V_lang = torch.linalg.svd(E_white, full_matrices=False)
            
            # 提取最大奇异值对应的特征方向作为噪声基底 B
            B = V_lang[:1, :].t() # 形状 [768, 1]
            
            # 构建正交投影算子 P = I - B * B^T
            I = torch.eye(dim, device=self.device)
            P = I - torch.mm(B, B.t())
            
            # 执行投影，滤除语言噪声结构
            E_pure = torch.mm(E_white, P)
            
            # =========================================================
            # 阶段 3: 本征基提取与流形压缩 (Truncated SVD)
            # =========================================================
            # 对纯净知识点云再次进行 SVD 分解提取核心骨架
            U_pure, S_pure, V_pure = torch.linalg.svd(E_pure, full_matrices=False)
            
            # 根据 mask_ratio 计算需要保留的本征秩 (Rank) r
            # mask_ratio 越高，保留的奇异值越少，压缩强度越大
            r = max(1, int(seq_len * (1.0 - mask_ratio)))
            r = min(r, dim) 
            
            # 重构压缩后的本征流形矩阵 (仅使用前 r 个基)
            E_compressed = torch.mm(U_pure[:, :r], torch.mm(torch.diag(S_pure[:r]), V_pure[:r, :]))
            
            # =========================================================
            # 阶段 4: 逆映射与张量恢复 (Inverse Transformation)
            # =========================================================
            # 核心步骤：必须将其反向映射回原始模型能理解的坐标系中，
            # 否则完全白化的向量输入 Transformer 会导致解码器崩溃。
            E_final = torch.mm(E_compressed, W_inv) + mean
            
            # 恢复 Batch 维度 -> [1, Seq_len, 768]
            modified_outputs = E_final.unsqueeze(0)
        
        # 构建日志信息，方便在 qa_manager.py 中追踪压缩效果
        log_msg = f"[ISE 处理完成] Token长度: {seq_len} | 提取本征秩: {r}/{seq_len} | 目标Mask率: {mask_ratio}"
        
        return modified_outputs, log_msg