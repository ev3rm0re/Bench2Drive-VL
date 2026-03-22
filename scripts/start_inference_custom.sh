#!/bin/bash

# Must set CARLA_ROOT explicitly for evaluation scripts
export CARLA_ROOT=/workspace/CARLA_0.9.15
export PYTHONPATH=${CARLA_ROOT}/PythonAPI/carla:${PYTHONPATH}
export PYTHONPATH=leaderboard:${PYTHONPATH}
export PYTHONPATH=leaderboard/team_code:${PYTHONPATH}
export PYTHONPATH=scenario_runner:${PYTHONPATH}
export PYTHONPATH=B2DVL_Adapter:${PYTHONPATH}
export SCENARIO_RUNNER_ROOT=scenario_runner

export EXTERNAL_CARLA=1 # Manually start CARLA server
# OpenDRIVE mesh tuning (helps remove raised side walls on custom .xodr maps)
export CARLA_ODR_VERTEX_DISTANCE=1.0
export CARLA_ODR_MAX_ROAD_LENGTH=50.0
export CARLA_ODR_WALL_HEIGHT=0.0
export CARLA_ODR_ADDITIONAL_WIDTH=0.0
export CARLA_ODR_SMOOTH_JUNCTIONS=1
export CARLA_ODR_ENABLE_MESH_VISIBILITY=1
# Strictly follow VLA direct-action outputs.
export ENABLE_VLA_DIRECT_ACTION=1
export STRICT_VLA_CONTROL=1
export VLA_DIRECT_ACTION_MAX_DEVIATION_DEG=20
export CUSTOM_MAX_SPEED_MPS=1.5
BASE_PORT=2000 # CARLA port
BASE_TM_PORT=8000 # CARLA traffic manager port
BASE_ROUTES=./custom_route # path to your route xml
TEAM_AGENT=leaderboard/team_code/data_agent.py # path to your agent, in B2DVL, the agent is fixed, so don't modify this
BASE_CHECKPOINT_ENDPOINT=./my_checkpoint # path to the checkpoint file with saves sceanario running process and results. 
# If not exist, it will be automatically created.
SAVE_PATH=./eval_v1/ # the directory where seonsor data is saved.
GPU_RANK=3 # CARLA GPU, keep different from VLM GPU to avoid render timeout
VLM_CONFIG=./configs/vlm_config_alpamayo.json
HOST="127.0.0.1"
PORT=$BASE_PORT
TM_PORT=$BASE_TM_PORT
ROUTES="${BASE_ROUTES}.xml"
CHECKPOINT_ENDPOINT="${BASE_CHECKPOINT_ENDPOINT}.json"
export RESET_CHECKPOINT=${RESET_CHECKPOINT:-1}
if [ "$RESET_CHECKPOINT" = "1" ] && [ -f "$CHECKPOINT_ENDPOINT" ]; then
	rm -f "$CHECKPOINT_ENDPOINT"
fi
export MINIMAL=0 # if MINIMAL > 0, DriveCommenter takes control of the ego vehicle,
# and vlm server is not needed
export EARLY_STOP=80 # When getting baseline data, we used a 80s early-stop to avoid wasting time on failed scenarios. You can delete this line to disable early-stop. 
bash leaderboard/scripts/run_evaluation.sh $PORT $TM_PORT 1 $ROUTES $TEAM_AGENT "." $CHECKPOINT_ENDPOINT $SAVE_PATH "null" $GPU_RANK $VLM_CONFIG $HOST