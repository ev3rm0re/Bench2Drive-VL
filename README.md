<p align="center"> <img src="./assets/Bench2Drive-VL.png" alt="B2DVL Header" style="max-width: 100%;" /> </p> <h2 align="center"><strong>Full-Stack Software for Closed-Loop Autonomous Driving with Vision Language Models</strong></h2><p align="center">Powered by the <code>DriveCommenter</code> expert model and CARLA simulator. </p> <p align="center"> <a href="https://thinklab-sjtu.github.io/Bench2Drive-VL/">📄 Document</a> | <a href="https://huggingface.co/datasets/Telkwevr/Bench2Drive-VL-base">📁 Dataset</a> </p>

## 🧠 Overview

**Bench2Drive-VL** is a full-stack software suite designed to accelerate the development and evaluation of Vision-Language Models for Autonomous Driving (**VLM4AD**).  
It supports the entire lifecycle—from dataset generation and annotation, model training, to closed-loop evaluation in simulation environments—focusing on modularity, reproducibility, and usability for both research and real-world applications.

It includes:

- 📦 **Annotated Dataset & Dynamic Label Annotator**  
  Automatically generates frame-level VQA pairs for both low-level perception (objects, signs, lanes) and high-level reasoning using the `DriveCommenter` module.

- 🧩 **Graph-based Reasoning & Chain-of-Thought Tools**  
  Define, visualize, and test custom graph-of-thought pipelines to guide VLMs through reasoning tasks.

- 🖼️ **Visualization & Human-in-the-loop GUIs**  
  Interactive GUIs to explore model predictions, revise annotations, and add new QA samples through visual debugging interfaces.

- 🤖 **Multimodal Baselines & Inference Tools**  
  Built-in models supporting diverse input formats (e.g., RGB, bounding boxes, language prompts) for fast prototyping and benchmarking.

- 🔄 **Closed-Loop Evaluation Benchmark in CARLA**  
  Seamless integration with the CARLA simulator and web-based APIs allows realistic, interactive, and scalable evaluation of VLMs in autonomous driving scenarios.

- 📦 **Docker Support & Detailed Documentation**  
  Easy-to-use Docker environments and thorough setup guides ensure a smooth installation and testing experience across platforms.

> 🐳 Docker support is on the way...

<p align="center">
  <img src="./assets/abstract_module.png" alt="B2DVL Modules" style="max-width: 80%;" />
</p>

---

## ⚙️ Getting Started

### 🔧 1. Environment Setup

#### Install CARLA

```bash
mkdir carla && cd carla
wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.15.tar.gz
tar -xvf CARLA_0.9.15.tar.gz
cd Import
wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/AdditionalMaps_0.9.15.tar.gz
cd .. && bash ImportAssets.sh
```

Set environment variables:

```bash
export CARLA_ROOT=YOUR_CARLA_PATH
echo "$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg" >> YOUR_CONDA_PATH/envs/YOUR_CONDA_ENV_NAME/lib/python3.7/site-packages/carla.pth
```

#### Write `env.sh`

```bash
export CARLA_ROOT=/path/to/your/carla
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg

export WORK_DIR=/path/to/this/repo
export PYTHONPATH=$PYTHONPATH:${WORK_DIR}/scenario_runner:${WORK_DIR}/leaderboard:${WORK_DIR}/B2DVL_Adapter

export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner
export LEADERBOARD_ROOT=${WORK_DIR}/leaderboard

export VQA_GEN=1
export STRICT_MODE=1
```

#### Activate Environment

```bash
source ./env.sh
```

---

## 🚀 Run Bench2Drive-VL

### 🧪 Closed-Loop Inference

1. **Write VLM config** ([example](./docs/qids.md)):

<details>
<summary>Click to view example</summary>

```jsonc
{
  "TASK_CONFIGS": {
    "FRAME_PER_SEC": 10
  },
  "INFERENCE_BASICS": {
    "INPUT_WINDOW": 1,
    "USE_ALL_CAMERAS": false,
    "USE_BEV": false,
    "NO_HISTORY_MODE": false
  },
  "CHAIN": {
    "NODE": [19, 15, 7, 24, 13, 47, 8, 43, 50],
    "EDGE": {
      "19": [24, 13, 8],
      "15": [7, 8],
      "7": [8],
      "24": [13, 47],
      "13": [47, 8, 43],
      "47": [8],
      "8": [43],
      "43": [50],
      "50": []
    },
    "INHERIT": {
      "19": [43, 7],
      "15": [7]
    },
    "USE_GT": [24]
  },
  "CONTROL_RATE": 2.0,
  "MODEL_NAME": "api",
  "MODEL_PATH": "../model_zoo/your_model",
  "GPU_ID": 0,
  "PORT": 7023,
  "IN_CARLA": true,
  "USE_BASE64": true,
  "NO_PERC_INFO": false
}
```
</details>

2. **Startup script**

```bash
export MINIMAL=0
bash leaderboard/scripts/run_evaluation.sh $PORT $TM_PORT 1 $ROUTES $TEAM_AGENT "." $CHECKPOINT_ENDPOINT $SAVE_PATH "null" $GPU_RANK $VLM_CONFIG
```

3. **Start VLM server (if not minimal)**

```bash
python ./B2DVL_Adapter/web_interact_app.py --config /path/to/your/vlm_config.json
```

4. **Launch main module**

```bash
bash ./startup.sh
```

---

### 📤 Generate VQAs from Dataset

1. **Write script under `./B2DVL-Adapter`:**

```bash
export SUBSET=0
export STRICT_MODE=1
export SUBSET_PATH=./subset_0.txt
export PROCESSED_PATH=./processed_paths_0.txt
export CACHE_PATH=./.worker_0_cache

python ./drive_commenter_main.py --data-directory=/path/to/Bench2Drive/dataset \
  --output-graph-directory=./outgraph \
  --path-maps=${CARLA_ROOT}/CarlaUE4/Content/Carla/Maps \
  --worker-count=1
```

2. **Run script**

```bash
cd ./B2DVL-Adapter
bash ./your_startup_script.sh
```

---

### 🔁 Open-Loop Inference

1. **Write config file**

<details>
<summary>Click to view config example</summary>

```jsonc
{
  "TASK_CONFIGS": {
    "INFER_SUBSET": false,
    "USE_CHECKPOINT": true,
    "SUBSET_FILE": "./infer_configs/subset.txt",
    "CHECKPOINT_FILE": "./infer_configs/finished_scenarios.txt",
    "ENTRY_EXIT_FILE": "./infer_configs/entry_exits.json",
    "FRAME_PER_SEC": 10
  },
  "INFERENCE_BASICS": {
    "INPUT_WINDOW": 1,
    "CONVERSATION_WINDOW": 2,
    "USE_ALL_CAMERAS": true,
    "NO_HISTORY_MODE": false,
    "APPEND_QUESTION": true,
    "APPENDIX_FILE": "./infer_configs/append_questions.json"
  },
  "CHAIN": {
    "NODE": [43, 50],
    "EDGE": {
      "43": [50],
      "50": []
    },
    "INHERIT": {
      "19": [43, 7],
      "15": [7]
    },
    "USE_GT": []
  }
}
```

</details>

2. **Run inference**

```bash
cd ./B2DVL_Adapter
python inference.py \
  --model Qwen2.5VL \
  --model_path /path/to/Qwen2.5VL-3B-Instruct \
  --config_dir /path/to/your_infer_config.json \
  --image_dir /path/to/Bench2Drive/dataset \
  --vqa_dir /path/to/vqa/dataset \
  --num_workers 4 \
  --out_dir ./infer_outputs
```

---

## 📊 Evaluation

1. **Use your own LLM API**

Create `mytoken.py` under `./B2DVL-Adapter`:

```python
DEEPSEEK_TOKEN = [
  "your-token-1",
  "your-token-2"
]
DEEPSEEK_URL = "https://api.deepseek.com/v1"
```

2. **Write config**

<details>
<summary>Click to view config example</summary>

```jsonc
{
  "EVAL_SUBSET": true,
  "USE_CHECKPOINT": false,
  "SUBSET_FILE": "./eval_configs/subset.txt",
  "CHECKPOINT_FILE": "./eval_configs/finished_scenarios.txt",
  "INFERENCE_RESULT_DIR": "./infer_results",
  "B2D_DIR": "/path/to/Bench2Drive/dataset",
  "ORIGINAL_VQA_DIR": "../Carla_Chain_QA/carla_vqa_gen/vqa_dataset/outgraph",
  "FRAME_PER_SEC": 10,
  "LOOK_FUTURE": false
}
```

</details>

3. **Run evaluation**

```bash
python eval.py \
  --config_dir ./path/to/eval_config.json \
  --num_workers 4 \
  --out_dir ./eval_outputs
```

---

## 📄 License

All assets and code are licensed under **CC-BY-NC-ND**, unless specified otherwise.
