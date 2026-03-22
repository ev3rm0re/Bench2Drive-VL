import json
import xml.etree.ElementTree as ET
from xml.dom import minidom
import glob
import math

def get_euler_from_matrix(matrix):
    # pitch is around Y, roll is around X, yaw is around Z in Unreal?
    pass # we can just use the rotation array if provided, or parse matrix.

def main():
    traj_path = "cc8c0bf57f984915a77078b10eb33198/trajectory.json"
    scene_path = "cc8c0bf57f984915a77078b10eb33198/scene.json"
    
    with open(traj_path, 'r') as f:
        traj = json.load(f)
        
    with open(scene_path, 'r') as f:
        scene = json.load(f)
        
    town_name = scene.get('map_name', 'Town01')
    
    frames = traj['frames']
    start_frame = frames[0]
    end_frame = frames[-1]
    
    def get_pose(frame):
        # frame['transform'] is 4x4 matrix
        mat = frame['transform']
        x = mat[0][3]
        y = mat[1][3]
        z = mat[2][3]
        # CARLA uses left-handed coords (often from Right-handed conversion)
        # But wait, DriveStudio JSON is already in CARLA Coordinates? 
        # DriveStudio generated it.
        # Check rotation
        rot = frame['rotation'] # might be [pitch, roll, yaw] or similar
        return x, y, z, rot[0], rot[1], rot[2]

    # ... generate xml
