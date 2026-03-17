import torch
from transformers import GPT2Tokenizer, GPT2Model

class MyCustomCompressor:
    def __init__(self):
        # 1. 确保设备与推理机一致
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 2. 加载基础组件
        # 注意：这里加载的是 GPT2Model（用于提取向量），而不是原本的 LMHeadModel
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2Model.from_pretrained("gpt2").to(self.device)
        self.model.eval()

    def compress(self, segments, mask_ratio):
        """
        验证版逻辑：将文字碎片转为向量，但不做魔改，原样返回。
        """
        # 1. 将碎片列表合并为完整字符串
        full_text = "".join(segments)
        
        # 2. 将字符串转为 Token IDs
        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)
        
        # 3. 通过词嵌入层（Embedding Layer）获取向量
        with torch.no_grad():
            # 得到的 outputs 形状为 [1, 序列长度, 768]
            outputs = self.model.get_input_embeddings()(inputs.input_ids)
        
        # 4. 返回魔改向量（此处为原向量）和占位日志
        # 确保第一个返回值是 torch.Tensor
        return outputs, ["standard_vector_bridge_active"]