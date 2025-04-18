
from .twod.datapipes.shallowwater2d import (
    onestep_test_datapipe_2day_vel,
    onestep_test_datapipe_2day_vort,
    onestep_valid_datapipe_2day_vel,
    onestep_valid_datapipe_2day_vort,
    train_datapipe_2day_vel,
    train_datapipe_2day_vort,
    trajectory_test_datapipe_2day_vel,
    trajectory_test_datapipe_2day_vort,
    trajectory_valid_datapipe_2day_vel,
    trajectory_valid_datapipe_2day_vort,
)

DATAPIPE_REGISTRY = {}

DATAPIPE_REGISTRY["ShallowWater2DVel-2Day"] = {}
DATAPIPE_REGISTRY["ShallowWater2DVel-2Day"]["train"] = train_datapipe_2day_vel
DATAPIPE_REGISTRY["ShallowWater2DVel-2Day"]["valid"] = [
    onestep_valid_datapipe_2day_vel,
    trajectory_valid_datapipe_2day_vel,
]
DATAPIPE_REGISTRY["ShallowWater2DVel-2Day"]["test"] = [
    onestep_test_datapipe_2day_vel,
    trajectory_test_datapipe_2day_vel,
]

DATAPIPE_REGISTRY["ShallowWater2DVort-2Day"] = {}
DATAPIPE_REGISTRY["ShallowWater2DVort-2Day"]["train"] = train_datapipe_2day_vort
DATAPIPE_REGISTRY["ShallowWater2DVort-2Day"]["valid"] = [
    onestep_valid_datapipe_2day_vort,
    trajectory_valid_datapipe_2day_vort,
]
DATAPIPE_REGISTRY["ShallowWater2DVort-2Day"]["test"] = [
    onestep_test_datapipe_2day_vort,
    trajectory_test_datapipe_2day_vort,
]
