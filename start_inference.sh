#!/bin/bash
export EXTERNAL_CARLA=1 # Manually start CARLA server
BASE_PORT=2000 # CARLA port
BASE_TM_PORT=8000 # CARLA traffic manager port
BASE_ROUTES=./leaderboard/data/drivetransformer_bench2drive_dev10 # path to your route xml
TEAM_AGENT=leaderboard/team_code/data_agent.py # path to your agent, in B2DVL, the agent is fixed, so don't modify this
BASE_CHECKPOINT_ENDPOINT=./my_checkpoint # path to the checkpoint file with saves sceanario running process and results. 
# If not exist, it will be automatically created.
SAVE_PATH=./eval_v1/ # the directory where seonsor data is saved.
GPU_RANK=2 # CARLA GPU, keep different from VLM GPU to avoid render timeout
VLM_CONFIG=./configs/vlm_config.json
HOST="127.0.0.1"
PORT=$BASE_PORT
TM_PORT=$BASE_TM_PORT
ROUTES="${BASE_ROUTES}.xml"
CHECKPOINT_ENDPOINT="${BASE_CHECKPOINT_ENDPOINT}.json"
export MINIMAL=0 # if MINIMAL > 0, DriveCommenter takes control of the ego vehicle,
# and vlm server is not needed
export EARLY_STOP=80 # When getting baseline data, we used a 80s early-stop to avoid wasting time on failed scenarios. You can delete this line to disable early-stop. 
bash leaderboard/scripts/run_evaluation.sh $PORT $TM_PORT 1 $ROUTES $TEAM_AGENT "." $CHECKPOINT_ENDPOINT $SAVE_PATH "null" $GPU_RANK $VLM_CONFIG $HOST