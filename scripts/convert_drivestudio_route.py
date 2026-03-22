import json
import xml.etree.ElementTree as ET
from xml.dom import minidom

def matrix_to_carla(mat, rot):
    # Depending on how the DriveStudio data is formatted (usually already in Carla format)
    # Pitch, Roll, Yaw
    pitch = rot[0]
    roll = rot[1]
    yaw = rot[2]
    x = mat[0][3]
    y = mat[1][3]
    z = mat[2][3] + 0.5 # slightly elevate so car doesn't get stuck in road
    return x, y, z, pitch, yaw, roll

def main():
    with open('cc8c0bf57f984915a77078b10eb33198/scene.json', 'r') as f:
        scene = json.load(f)
    town_name = scene.get('map_name', 'Town01') # Usually 'singapore-onenorth' or similar
    
    with open('cc8c0bf57f984915a77078b10eb33198/trajectory.json', 'r') as f:
        traj = json.load(f)
        
    frames = traj['frames']
    start_frame = frames[0]
    end_frame = frames[-1]
    
    # We may need a bit early stop or exact end
    # For routes.xml
    root = ET.Element("routes")
    route_xml = ET.SubElement(root, "route", id="0", town=town_name)
    ET.SubElement(route_xml, "weather", cloudiness="0.0", precipitation="0.0", precipitation_deposits="0.0", wind_intensity="0.0", sun_azimuth_angle="0.0", sun_altitude_angle="90.0", fog_density="0.0", wetness="0.0", dropness="0.0")
    
    # Add waypoints. By minimum, leaderboad only needs start and end.
    waypoint_start = ET.SubElement(route_xml, "waypoint")
    x, y, z, p, yaw, r = matrix_to_carla(start_frame['transform'], start_frame['rotation'])
    waypoint_start.set('x', f"{x:.4f}")
    waypoint_start.set('y', f"{y:.4f}")
    waypoint_start.set('z', f"{z:.4f}")
    waypoint_start.set('pitch', f"{p:.4f}")
    waypoint_start.set('yaw', f"{yaw:.4f}")
    waypoint_start.set('roll', f"{r:.4f}")
    
    waypoint_end = ET.SubElement(route_xml, "waypoint")
    x, y, z, p, yaw, r = matrix_to_carla(end_frame['transform'], end_frame['rotation'])
    waypoint_end.set('x', f"{x:.4f}")
    waypoint_end.set('y', f"{y:.4f}")
    waypoint_end.set('z', f"{z:.4f}")
    waypoint_end.set('pitch', f"{p:.4f}")
    waypoint_end.set('yaw', f"{yaw:.4f}")
    waypoint_end.set('roll', f"{r:.4f}")

    xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
    with open("custom_route.xml", "w") as f:
        f.write(xml_str)
    
    print("Generated custom_route.xml")

if __name__ == '__main__':
    main()
