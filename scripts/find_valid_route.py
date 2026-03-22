import carla
import json
from agents.navigation.global_route_planner import GlobalRoutePlanner
import networkx as nx

client = carla.Client('127.0.0.1', 2000)
world = client.get_world()
m = world.get_map()
grp = GlobalRoutePlanner(m, 2.0)

try:
    with open('/workspace/Bench2Drive-VL/route.json', 'r') as f:
        data = json.load(f)
except Exception as e:
    print("Could not load route.json:", e)
    data = []

found_any = False
for obj in data:
    if len(obj.get('waypoints', [])) < 10:
        continue
    
    print(f"Checking object {obj.get('id')}, length {len(obj['waypoints'])}")
    pts = []
    for w in obj['waypoints']:
        loc = carla.Location(x=w['position'][0], y=w['position'][1], z=0.5)
        wp = m.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        
        # fallback to lane 1 if lane -1 not in edges
        if wp.road_id in grp._road_id_to_edge and wp.section_id in grp._road_id_to_edge[wp.road_id] and wp.lane_id not in grp._road_id_to_edge[wp.road_id][wp.section_id]:
            if wp.lane_id < 0:
                alt_wp = wp.get_left_lane()
                while alt_wp and alt_wp.lane_id < 0:
                    alt_wp = alt_wp.get_left_lane()
                if alt_wp: wp = alt_wp
            else:
                alt_wp = wp.get_right_lane()
                while alt_wp and alt_wp.lane_id > 0:
                    alt_wp = alt_wp.get_right_lane()
                if alt_wp: wp = alt_wp
        
        pts.append(wp.transform.location)

    n_start = grp._localize(pts[0])
    n_end = grp._localize(pts[-1])
    
    if n_start and n_end:
        for u in n_start:
            for v in n_end:
                if nx.has_path(grp._graph, u, v):
                    print(f"   FOUND path for object! start: {pts[0]} (node {u}), end: {pts[-1]} (node {v})")
                    found_any = True
                    break

if not found_any:
    print("No valid route found in route.json data.")
    
    # Let's try to just generate a random valid route to use for testing
    print("Generating a random valid route instead...")
    topology = m.get_topology()
    if topology:
        wp1 = topology[0][0]
        wp2 = topology[-1][0]
        n1 = grp._localize(wp1.transform.location)
        n2 = grp._localize(wp2.transform.location)
        print("Using random topology waypoints:")
        print("Start:", wp1.transform.location)
        print("End:", wp2.transform.location)
