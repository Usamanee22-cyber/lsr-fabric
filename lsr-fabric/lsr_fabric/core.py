import torch
import torch.nn as nn

class LSRGateway(nn.Module):
    """
    Gateway อัจฉริยะประเมินว่า Token นี้ยากหรือง่าย 
    และทำหน้าที่เป็น 'Adapter' ตอน Fine-tuning ไปในตัว
    """
    def __init__(self, hidden_size, threshold=0.15, mode="inference"):
        super().__init__()
        self.gateway_layer = nn.Linear(hidden_size, 1)
        self.threshold = threshold
        self.mode = mode # mode สามารถเป็น: 'inference', 'finetune', 'scratch'
        
        # คลังเก็บพารามิเตอร์เสริม (เปรียบเสมือน LoRA ขนาดจิ๋วฝังในตัวสับราง)
        self.adaptation_layer = nn.Linear(hidden_size, hidden_size, bias=False)
        # ตั้งค่าเริ่มต้นให้เป็นศูนย์เพื่อไม่ให้กระทบโมเดลเดิมในตอนแรก
        nn.init.zeros_(self.adaptation_layer.weight)

    def forward(self, hidden_states):
        # 1. คำนวณความมั่นใจ
        confidence_score = torch.sigmoid(self.gateway_layer(hidden_states))
        should_skip = confidence_score.mean().item() < self.threshold
        
        # 2. ปรับพฤติกรรมตามโหมด (ตอบโจทย์กติกาเพิ่มเติม)
        if self.mode == "finetune":
            # ในโหมด Fine-tuning จะส่งสารอาหาร (Gradients) ผ่าน Adaptation Layer จิ๋วนี้
            hidden_states = hidden_states + self.adaptation_layer(hidden_states)
            
        return should_skip, hidden_states

class LSRFabricWrapper(nn.Module):
    """
    ตัวครอบชั้น Layer เดิม ไม่ว่าจะเป็นโมเดลเก่า หรือโมเดลสร้างใหม่จากศูนย์
    """
    def __init__(self, original_layer, hidden_size, threshold=0.15, mode="inference"):
        super().__init__()
        self.original_layer = original_layer
        self.gateway = LSRGateway(hidden_size, threshold, mode)

    def forward(self, hidden_states, *args, **kwargs):
        # รันผ่านช่องประตู Gateway ก่อน
        should_skip, hidden_states = self.gateway(hidden_states)
        
        # โหมดสร้างจากศูนย์ (scratch) จะไม่ Skip ทันทีแต่จะใช้ความมั่นใจไปคุมน้ำหนักลอส
        if self.gateway.mode == "scratch":
            return self.original_layer(hidden_states, *args, **kwargs)
            
        # โหมด Inference / Fine-tune ทั่วไป
        if should_skip and self.gateway.mode == "inference":
            return (hidden_states,) # ⚡ Skip! ทางด่วนพิเศษ ไม่คิด Compute ใน Layer นี้
        else:
            return self.original_layer(hidden_states, *args, **kwargs) # 🐢 คำนวณปกติ