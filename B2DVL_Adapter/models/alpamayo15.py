import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from .VLMInterface import VLMInterface
from io_utils import load_json_gz


class Alpamayo15Interface(VLMInterface):
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
        print(f"Initializing Alpamayo-1.5-10B on GPU {gpu_id}...")

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
        self.dtype = torch.bfloat16

        try:
            from alpamayo1_5 import helper
            from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
        except Exception as e:
            raise RuntimeError(
                "Failed to import alpamayo1_5. Install from official repo first. "
                f"Details: {e}"
            )

        self._helper = helper
        repo_or_path = self.model_path if self.model_path else "nvidia/Alpamayo-1.5-10B"
        # Default to sdpa so the runtime can work even when flash-attn is unavailable.
        attn_impl = os.environ.get("ALPAMAYO_ATTN_IMPL", "eager")
        self.model = Alpamayo1_5.from_pretrained(
            repo_or_path,
            dtype=self.dtype,
            attn_implementation=attn_impl,
        ).to(self.device)
        self.processor = helper.get_processor(self.model.tokenizer)
        self.model.eval()

        print(f"Alpamayo-1.5-10B loaded on GPU {gpu_id} successfully")

    def _extract_navigation_command(self, text: str) -> str:
        if not isinstance(text, str):
            return "FOLLOW_LANE"

        upper_text = text.upper()
        if "CHANGE" in upper_text and "LEFT" in upper_text:
            return "CHANGE_LANE_LEFT"
        if "CHANGE" in upper_text and "RIGHT" in upper_text:
            return "CHANGE_LANE_RIGHT"
        if "DEVIATE" in upper_text and "LEFT" in upper_text:
            return "DEVIATE_LEFT"
        if "DEVIATE" in upper_text and "RIGHT" in upper_text:
            return "DEVIATE_RIGHT"
        if "TURN" in upper_text and "LEFT" in upper_text:
            return "TURN_LEFT"
        if "TURN" in upper_text and "RIGHT" in upper_text:
            return "TURN_RIGHT"
        if "STRAIGHT" in upper_text:
            return "GO_STRAIGHT"
        return "FOLLOW_LANE"

    def _collect_image_paths(self, bubble, current_frame: int) -> List[str]:
        images_dict = bubble.get_full_images()
        frame_ids = sorted(images_dict.keys())
        chosen_paths: List[str] = []

        for frame_id in frame_ids:
            if frame_id > current_frame:
                continue
            if self.no_history and frame_id != current_frame:
                continue

            frame_images = images_dict.get(frame_id, {})
            if self.in_carla:
                if self.use_bev and "ANNO_BEV" in frame_images:
                    chosen_paths.append(frame_images["ANNO_BEV"])
                elif self.use_all_cameras:
                    for key in ["CAM_FRONT_CONCAT", "CAM_BACK_CONCAT"]:
                        if key in frame_images:
                            chosen_paths.append(frame_images[key])
                elif "ANNO_CAM_FRONT" in frame_images:
                    chosen_paths.append(frame_images["ANNO_CAM_FRONT"])
            else:
                if self.use_all_cameras:
                    for key in ["CAM_FRONT_CONCAT", "CAM_BACK_CONCAT"]:
                        if key in frame_images:
                            chosen_paths.append(frame_images[key])
                elif "CAM_FRONT" in frame_images:
                    chosen_paths.append(frame_images["CAM_FRONT"])

        return chosen_paths

    @staticmethod
    def _rotation_matrix_from_carla_rotation(rotation: List[float]) -> np.ndarray:
        # Bench2Drive uses [pitch, roll, yaw] in most modules where yaw is index 2.
        pitch = np.deg2rad(float(rotation[0]))
        roll = np.deg2rad(float(rotation[1]))
        yaw = np.deg2rad(float(rotation[2]))

        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll), np.sin(roll)
        cy, sy = np.cos(yaw), np.sin(yaw)

        r_z = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        r_y = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float32)
        r_x = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float32)
        return r_z @ r_y @ r_x

    @staticmethod
    def _find_ego_bbox(anno_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        for actor in anno_data.get("bounding_boxes", []):
            if actor.get("class") == "ego_vehicle":
                return actor
        return None

    def _build_history_from_annos(self, anno_file: str, steps: int = 16) -> Tuple[torch.Tensor, torch.Tensor]:
        if not anno_file or not os.path.exists(anno_file):
            raise ValueError(f"Invalid anno file: {anno_file}")

        anno_dir = os.path.dirname(anno_file)
        frame_str = os.path.basename(anno_file).split(".")[0]
        current_idx = int(frame_str)

        hist_ids = list(range(max(0, current_idx - steps + 1), current_idx + 1))

        positions: List[np.ndarray] = []
        rotations: List[np.ndarray] = []

        for idx in hist_ids:
            cand_gz = os.path.join(anno_dir, f"{idx:05d}.json.gz")
            cand_json = os.path.join(anno_dir, f"{idx:05d}.json")
            path = cand_gz if os.path.exists(cand_gz) else cand_json
            if not os.path.exists(path):
                continue

            anno = load_json_gz(path)
            ego = self._find_ego_bbox(anno) if isinstance(anno, dict) else None
            if ego is None:
                continue

            positions.append(np.asarray(ego["location"], dtype=np.float32))
            rotations.append(self._rotation_matrix_from_carla_rotation(ego["rotation"]))

        if len(positions) == 0:
            raise ValueError("No ego history in annotation sequence")

        # Alpamayo trajectory tokenization expects a stable history length.
        # For early frames, pad by repeating the earliest available state.
        if len(positions) < steps:
            pad_count = steps - len(positions)
            first_p = positions[0]
            first_r = rotations[0]
            positions = [first_p.copy() for _ in range(pad_count)] + positions
            rotations = [first_r.copy() for _ in range(pad_count)] + rotations
        elif len(positions) > steps:
            positions = positions[-steps:]
            rotations = rotations[-steps:]

        p0 = positions[-1]
        r0 = rotations[-1]
        r0_inv = r0.T

        pos_local = []
        rot_local = []
        for p, r in zip(positions, rotations):
            pos_local.append(r0_inv @ (p - p0))
            rot_local.append(r0_inv @ r)

        pos_local = np.stack(pos_local, axis=0)
        rot_local = np.stack(rot_local, axis=0)

        ego_history_xyz = torch.from_numpy(pos_local).float().unsqueeze(0).unsqueeze(0)
        ego_history_rot = torch.from_numpy(rot_local).float().unsqueeze(0).unsqueeze(0)
        return ego_history_xyz, ego_history_rot

    @staticmethod
    def _images_to_frame_tensor(image_paths: List[str]) -> torch.Tensor:
        frames = []
        for path in image_paths:
            img = Image.open(path).convert("RGB")
            # np.asarray(PIL.Image) can be non-writable; make a writable copy.
            arr = np.asarray(img, dtype=np.uint8).copy()
            frames.append(torch.from_numpy(arr).permute(2, 0, 1).contiguous())

        if len(frames) == 0:
            raise ValueError("No input images found for Alpamayo")

        return torch.stack(frames, dim=0)

    def _infer_direction_key(self, traj_xy: np.ndarray, direction_hint: str) -> str:
        if traj_xy.shape[0] < 2:
            return "FOLLOW_LANE"

        start = traj_xy[0]
        end = traj_xy[-1]
        delta = end - start
        forward = float(delta[0])
        lateral = float(delta[1])

        idx_mid = min(10, traj_xy.shape[0] - 1)
        d_mid = traj_xy[idx_mid] - start
        heading = float(np.arctan2(d_mid[1], max(1e-3, d_mid[0])))

        lane_change_candidate = None
        if forward > 6.0 and abs(lateral) > 2.2 and abs(heading) < 0.28:
            lane_change_candidate = "CHANGE_LANE_LEFT" if lateral > 0 else "CHANGE_LANE_RIGHT"

        # Safety gate: only emit lane-change keys when route/nav hint explicitly asks for lane change.
        if lane_change_candidate is not None:
            if direction_hint in {"CHANGE_LANE_LEFT", "CHANGE_LANE_RIGHT"}:
                return lane_change_candidate
            return "DEVIATE_LEFT" if lateral > 0 else "DEVIATE_RIGHT"
        if heading > 0.35 and forward > 1.0:
            return "TURN_LEFT"
        if heading < -0.35 and forward > 1.0:
            return "TURN_RIGHT"
        if forward > 1.0 and 0.8 < abs(lateral) <= 2.2:
            return "DEVIATE_LEFT" if lateral > 0 else "DEVIATE_RIGHT"

        # Only use hint for intersection intent, never for lane-change override.
        if direction_hint in {"TURN_LEFT", "TURN_RIGHT", "GO_STRAIGHT"} and forward > 1.0:
            return direction_hint
        return "FOLLOW_LANE"

    def _trajectory_to_action(self, pred_xyz: torch.Tensor) -> Tuple[Dict[str, float], str, str, Dict[str, float]]:
        # Shape from official API: [B, n_set, n_sample, T, 3]
        traj = pred_xyz[0, 0, 0, :, :2].detach().float().cpu().numpy()
        t_len = traj.shape[0]

        idx_steer = min(8, t_len - 1)
        p_steer = traj[idx_steer]

        steer = np.arctan2(float(p_steer[1]), max(0.8, float(p_steer[0]))) / 0.8
        steer = float(np.clip(steer, -1.0, 1.0))

        diffs = np.diff(traj, axis=0)
        path_len = float(np.linalg.norm(diffs, axis=1).sum()) if len(diffs) > 0 else 0.0
        horizon = max((t_len - 1) * 0.1, 0.1)
        planned_speed = path_len / horizon
        forward_rate = float(traj[-1, 0] - traj[0, 0]) / horizon

        if planned_speed < 0.25 and forward_rate < 0.05:
            speed_key = "STOP"
            throttle = 0.0
            brake = 0.65
        elif planned_speed < 1.2:
            speed_key = "DECELERATE"
            throttle = 0.12
            brake = 0.0
        elif planned_speed < 4.0:
            speed_key = "KEEP"
            throttle = 0.35
            brake = 0.0
        else:
            speed_key = "ACCELERATE"
            throttle = 0.6
            brake = 0.0

        action = {
            "steer": float(np.clip(steer, -1.0, 1.0)),
            "throttle": float(np.clip(throttle, 0.0, 1.0)),
            "brake": float(np.clip(brake, 0.0, 1.0)),
        }
        metrics = {
            "planned_speed": planned_speed,
            "forward_rate": forward_rate,
            "path_len": path_len,
        }
        return action, speed_key, metrics

    def interact(self, bubble, conversation):
        torch.cuda.set_device(self.gpu_id)
        self.device = torch.device(f"cuda:{self.gpu_id}")

        qid = getattr(bubble, "qid", -1)
        direction_hint = self._extract_navigation_command(bubble.get_full_words())

        try:
            meta = bubble.transform if isinstance(bubble.transform, dict) else {}
            anno_file = meta.get("anno_file")
            current_frame = int(meta.get("frame_number", bubble.frame_number))

            image_paths = self._collect_image_paths(bubble, current_frame)
            frame_tensor = self._images_to_frame_tensor(image_paths)
            ego_history_xyz, ego_history_rot = self._build_history_from_annos(anno_file)

            # Keep official prompt style but avoid forcing route text from noisy question parsing.
            messages = self._helper.create_message(frame_tensor, use_nav_prompt=True)
            tokenized = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                continue_final_message=True,
                return_dict=True,
                return_tensors="pt",
            )

            model_inputs = {
                "tokenized_data": tokenized,
                "ego_history_xyz": ego_history_xyz,
                "ego_history_rot": ego_history_rot,
            }
            model_inputs = self._helper.to_device(model_inputs, self.device)

            torch.cuda.manual_seed_all(42)
            with torch.autocast("cuda", dtype=self.dtype):
                pred_xyz, pred_rot, extra = self.model.sample_trajectories_from_data_with_vlm_rollout(
                    data=model_inputs,
                    top_p=0.98,
                    temperature=0.6,
                    num_traj_samples=1,
                    max_generation_length=256,
                    return_extra=True,
                )

            action, speed_key, metrics = self._trajectory_to_action(pred_xyz)
            traj_xy = pred_xyz[0, 0, 0, :, :2].detach().float().cpu().numpy()
            direction_key = self._infer_direction_key(traj_xy, direction_hint)
            cot = ""
            if isinstance(extra, dict) and "cot" in extra:
                try:
                    cot_obj = extra["cot"]
                    if hasattr(cot_obj, "flatten"):
                        flat = cot_obj.flatten()
                        cot = str(flat[0]) if len(flat) > 0 else ""
                    else:
                        cot = str(cot_obj)
                except Exception:
                    cot = ""

            action_dict = {
                "type": "direct_action",
                "steer": action["steer"],
                "throttle": action["throttle"],
                "brake": action["brake"],
                "direction_key": direction_key,
                "speed_key": speed_key,
                "raw": cot,
            }

            output = (
                f"Direction Key = {direction_key}, Speed Key = {speed_key}. "
                f"Action Tokens: <steer_{action['steer']:.3f}><throttle_{action['throttle']:.3f}><brake_{action['brake']:.3f}>"
            )

            print(f"[Alpamayo-1.5-10B] COT: {cot}")
            print(f"[Alpamayo-1.5-10B] Metrics: {metrics}")
            print(f"[Alpamayo-1.5-10B] Output: {output}")
            if qid == 50:
                return action_dict
            return output

        except Exception as e:
            print(f"Alpamayo interact error: {e}")
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
        raise NotImplementedError("Alpamayo15Interface uses interact() in Bench2Drive runtime.")
