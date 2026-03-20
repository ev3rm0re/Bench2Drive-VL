export CARLA_ROOT=/workspace/CARLA_0.9.15
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI:${CARLA_ROOT}/PythonAPI/carla
# export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg

export WORK_DIR=/workspace/Bench2Drive-VL
export PYTHONPATH=$PYTHONPATH:${WORK_DIR}/scenario_runner:${WORK_DIR}/leaderboard:${WORK_DIR}/B2DVL_Adapter
export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner
export LEADERBOARD_ROOT=${WORK_DIR}/leaderboard