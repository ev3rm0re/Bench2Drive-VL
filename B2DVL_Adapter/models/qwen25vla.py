from .VLMInterface import VLMInterface
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import torch
import re

class Qwen25VLAInterface(VLMInterface):
    def initialize(self, gpu_id: int, use_all_cameras: bool, no_history: bool, 
                    input_window: int, frame_rate: int, model_path: str, use_bev: bool=False, 
                    in_carla: bool=False, use_base64: bool=False):
        print(f"Initializing Qwen2.5-VLA on GPU {gpu_id}...")
        self.gpu_id = gpu_id
        torch.cuda.set_device(self.gpu_id)
        self.device = torch.device(f"cuda:{self.gpu_id}")
        
        # 加载你微调过 Action Token 的 Qwen2.5-VL 权重
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16,
            device_map=self.device
        )
        self.model.eval()

    def inference(self, input_conversation, **kwargs):
        """
        VLA 推理逻辑：输入图像+简单指令 -> 直接输出动作
        """
        try:
            minimal_conversation = []
    
            # 2. 从框架传来的冗长数据中，把图片摘出来，文本全部丢弃或替换
            for msg in input_conversation:
                if msg.get("type") == "image":
                    minimal_conversation.append(msg)  # 原封不动保留多视角图片
                elif msg.get("type") == "text":
                    # 原始文本里其实包含了当前的高层导航指令（比如"Turn Left"）
                    # 你可以写一段正则把指令提取出来，或者直接替换为极简 Prompt
                    original_text = msg.get("text", "")
                    command = extract_command(original_text) # 你自定义的提取函数

                    minimal_prompt = f"Navigation command: {command}. Output the exact driving action tokens."
                    minimal_conversation.append({"type": "text", "text": minimal_prompt})
            # 1. 预处理输入 (通常在 Adapter 层已经构造好 messages)
            text = self.processor.apply_chat_template(
                minimal_conversation, tokenize=False, add_generation_prompt=True
            )
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(input_conversation)
            
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            # 2. 生成 VLA 动作 Tokens
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=64, do_sample=False)
                
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            decoded = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            action_dict = {
                "type": "direct_action",
                "steer": 0.0,
                "throttle": 0.0,
                "brake": 0.0
            }
            
            steer_match = re.search(r'<steer_([-\d\.]+)>', decoded)
            throttle_match = re.search(r'<throttle_([\d\.]+)>', decoded)
            brake_match = re.search(r'<brake_([\d\.]+)>', decoded)
            
            if steer_match:
                action_dict["steer"] = float(steer_match.group(1))
            if throttle_match:
                action_dict["throttle"] = float(throttle_match.group(1))
            if brake_match:
                action_dict["brake"] = float(brake_match.group(1))

            print(f"[VLA Output] Raw: {decoded} -> Parsed Action: {action_dict}")
            
            # 直接返回结构化的动作字典，而不是自然语言字符串
            return action_dict

        except Exception as e:
            print(f"VLA Inference Error: {e}")
            return {"type": "direct_action", "steer": 0.0, "throttle": 0.0, "brake": 1.0} # 报错时默认紧急刹车