import torch
from .core import LSRFabricWrapper

def apply_fabric(model, threshold=0.15, mode="inference"):
    """
    ฟังก์ชันวิเศษที่ใช้แฮกสถาปัตยกรรมโมเดลเดิมในหน่วยความจำ
    """
    # ค้นหาโมเดลในระดับ Layer (ซัพพอร์ตตระกูล Llama / Typhoon / Mistral)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
        hidden_size = model.config.hidden_size
        
        for i, layer in enumerate(layers):
            # เอาสถาปัตยกรรมใหม่ของเราไปครอบทับแต่ละ Layer เดิม
            layers[i] = LSRFabricWrapper(layer, hidden_size, threshold, mode)
            
        print(f"🎉 [LSR-Fabric] สวมสถาปัตยกรรมใหม่เสร็จสิ้นในโหมด: {mode}")
    else:
        raise ValueError("สถาปัตยกรรมโมเดลนี้ยังไม่ได้รับการสนับสนุนในแพ็กเกจเวอร์ชันแรก")
        
    # ปรับพฤติกรรมการอัปเดต Weight ตามกติกาพิเศษของคุณ
    if mode == "finetune":
        # ฟรีซโมเดลหลักทั้งหมด!
        for param in model.parameters():
            param.requires_grad = False
        # เปิดให้เทรนเฉพาะตัว Fabric เท่านั้น (ประหยัดแรมการ์ดจอ)
        for name, param in model.named_parameters():
            if "gateway" in name:
                param.requires_grad = True
                
    elif mode == "scratch":
        # ถ้าสร้างจากศูนย์ เปิดให้ทุกอย่างอัปเดตพร้อมกันหมดอย่างสมดุล
        for param in model.parameters():
            param.requires_grad = True
            
    return model