from .VLMInterface import VLMInterface
from .interact_utils import get_image_descriptions, get_carla_image_descriptions
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import torch
import re


def image_template(image_path, use_base64):
    return {
        "type": "image",
        "image": image_path if use_base64 else f"file://{image_path}",
    }


class Qwen25VLAInterface(VLMInterface):
    def initialize(
        self,
        gpu_id: int,
        use_all_cameras: bool,
        no_history: bool,
        input_window: int,
        frame_rate: int,
        model_path: str,
        use_bev: bool = False,
        in_carla: bool = False,
        use_base64: bool = False,
    ):
        print(f"Initializing Qwen2.5-VLA on GPU {gpu_id}...")
        self.in_carla = in_carla
        self.use_bev = use_bev
        self.use_all_cameras = use_all_cameras
        self.input_window = input_window
        self.no_history = no_history
        self.gpu_id = gpu_id
        self.model_path = model_path
        self.frame_rate = frame_rate
        self.use_base64 = use_base64

        torch.cuda.set_device(self.gpu_id)
        self.device = torch.device(f"cuda:{self.gpu_id}")

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )
        self.model.to(self.device)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(self.model_path)

        print(f"Qwen2.5-VLA (Patched) loaded on GPU {gpu_id} successfully")

    def _extract_command(self, text: str) -> str:
        if not isinstance(text, str):
            return "FOLLOW_LANE"

        upper_text = text.upper()

        token_candidates = [
            "FOLLOW_LANE",
            "CHANGE_LANE_LEFT",
            "CHANGE_LANE_RIGHT",
            "DEVIATE_LEFT",
            "DEVIATE_RIGHT",
            "GO_STRAIGHT",
            "TURN_LEFT",
            "TURN_RIGHT",
        ]
        for token in token_candidates:
            if token in upper_text:
                return token

        if "CHANGE" in upper_text and "LEFT" in upper_text:
            return "CHANGE_LANE_LEFT"
        if "CHANGE" in upper_text and "RIGHT" in upper_text:
            return "CHANGE_LANE_RIGHT"
        if "TURN" in upper_text and "LEFT" in upper_text:
            return "TURN_LEFT"
        if "TURN" in upper_text and "RIGHT" in upper_text:
            return "TURN_RIGHT"
        if "STRAIGHT" in upper_text:
            return "GO_STRAIGHT"

        return "FOLLOW_LANE"

    def _parse_action_tokens(self, text: str):
        action = {"steer": 0.0, "throttle": 0.0, "brake": 0.0}
        if not isinstance(text, str):
            return action

        steer_match = re.search(r"<\s*steer_\s*([-+]?\d*\.?\d+)\s*>", text, flags=re.IGNORECASE)
        throttle_match = re.search(r"<\s*throttle_\s*([-+]?\d*\.?\d+)\s*>", text, flags=re.IGNORECASE)
        brake_match = re.search(r"<\s*brake_\s*([-+]?\d*\.?\d+)\s*>", text, flags=re.IGNORECASE)

        if steer_match:
            val = float(steer_match.group(1))
            if abs(val) > 1.0:
                val /= 100.0
            action["steer"] = val
        if throttle_match:
            val = float(throttle_match.group(1))
            if val > 1.0:
                val /= 100.0
            action["throttle"] = val
        if brake_match:
            val = float(brake_match.group(1))
            if val > 1.0:
                val /= 100.0
            action["brake"] = val

        action["steer"] = max(-1.0, min(1.0, action["steer"]))
        action["throttle"] = max(0.0, min(1.0, action["throttle"]))
        action["brake"] = max(0.0, min(1.0, action["brake"]))
        return action

    def _action_to_keys(self, action, command_hint: str):
        if action["brake"] > 0.4:
            speed_key = "STOP"
        elif action["throttle"] > 0.6:
            speed_key = "ACCELERATE"
        elif action["throttle"] < 0.15:
            speed_key = "DECELERATE"
        else:
            speed_key = "KEEP"

        direction_key = command_hint if command_hint else "FOLLOW_LANE"
        if direction_key not in {
            "FOLLOW_LANE",
            "CHANGE_LANE_LEFT",
            "CHANGE_LANE_RIGHT",
            "DEVIATE_LEFT",
            "DEVIATE_RIGHT",
            "GO_STRAIGHT",
            "TURN_LEFT",
            "TURN_RIGHT",
        }:
            direction_key = "FOLLOW_LANE"

        return direction_key, speed_key

    def get_image_descriptions(self, images_dict, image_frame_list, start_frame, end_frame):
        if self.in_carla:
            return get_carla_image_descriptions(
                images_dict=images_dict,
                image_frame_list=image_frame_list,
                start_frame=start_frame,
                end_frame=end_frame,
                frame_rate=self.frame_rate,
                template_func=image_template,
                use_all_cameras=self.use_all_cameras,
                use_bev=self.use_bev,
                use_base64=self.use_base64,
            )
        return get_image_descriptions(
            images_dict=images_dict,
            image_frame_list=image_frame_list,
            start_frame=start_frame,
            end_frame=end_frame,
            frame_rate=self.frame_rate,
            template_func=image_template,
            use_all_cameras=self.use_all_cameras,
            use_base64=self.use_base64,
        )

    def interact(self, bubble, conversation):
        torch.cuda.set_device(self.gpu_id)
        self.device = torch.device(f"cuda:{self.gpu_id}")

        images_list = bubble.get_full_images()
        image_frame_list = sorted(images_list.keys())
        current_frame = bubble.frame_number

        input_conversation = []
        user_message = {"role": "user", "content": []}

        if len(image_frame_list) > 0:
            image_content, _ = self.get_image_descriptions(
                images_list,
                image_frame_list,
                current_frame - 1,
                current_frame,
            )
            user_message["content"].extend(image_content)

        qid = getattr(bubble, "qid", -1)
        full_text = bubble.get_full_words()
        command = self._extract_command(full_text)
        action_prompt = (
            "You are an autonomous driving action model. "
            "Based on the images and command, output only action tokens in this exact format: "
            "<steer_xxx><throttle_xxx><brake_xxx>. "
            f"Navigation command: {command}."
        )
        user_message["content"].append({"type": "text", "text": action_prompt})
        input_conversation.append(user_message)

        try:
            prompt = self.processor.apply_chat_template(
                input_conversation,
                tokenize=False,
                add_generation_prompt=True,
            )
            image_inputs, video_inputs = process_vision_info(input_conversation)

            inputs = self.processor(
                text=[prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=64,
                    do_sample=False,
                )

            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            decoded = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            action = self._parse_action_tokens(decoded)
            direction_key, speed_key = self._action_to_keys(action, command)

            action_dict = {
                "type": "direct_action",
                "steer": action["steer"],
                "throttle": action["throttle"],
                "brake": action["brake"],
                "direction_key": direction_key,
                "speed_key": speed_key,
                "raw": decoded,
            }

            output = (
                f"Direction Key = {direction_key}, Speed Key = {speed_key}. "
                f"Action Tokens: <steer_{action['steer']:.3f}><throttle_{action['throttle']:.3f}><brake_{action['brake']:.3f}>"
            )

            print(f"[Qwen2.5-VLA] Raw output: {decoded}")
            print(f"[Qwen2.5-VLA] Normalized output: {output}")
            if qid == 50:
                return action_dict
            return output

        except Exception as e:
            print(f"VLA interact error: {e}")
            fallback = {
                "type": "direct_action",
                "steer": 0.0,
                "throttle": 0.0,
                "brake": 1.0,
                "direction_key": "FOLLOW_LANE",
                "speed_key": "STOP",
                "raw": "",
            }
            if qid == 50:
                return fallback
            return "Direction Key = FOLLOW_LANE, Speed Key = STOP. Action Tokens: <steer_0.000><throttle_0.000><brake_1.000>"

    def inference(self, input_conversation, **kwargs):
        try:
            prompt = self.processor.apply_chat_template(
                input_conversation,
                tokenize=False,
                add_generation_prompt=True,
            )
            image_inputs, video_inputs = process_vision_info(input_conversation)

            inputs = self.processor(
                text=[prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=64,
                    do_sample=False,
                )

            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            decoded = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            action = self._parse_action_tokens(decoded)
            return {
                "type": "direct_action",
                "steer": action["steer"],
                "throttle": action["throttle"],
                "brake": action["brake"],
                "raw": decoded,
            }
        except Exception as e:
            print(f"VLA inference error: {e}")
            return {
                "type": "direct_action",
                "steer": 0.0,
                "throttle": 0.0,
                "brake": 1.0,
                "raw": "",
            }