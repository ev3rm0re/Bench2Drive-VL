import json
import carla
import random
import time
import math

def get_transform(mat):
    # From the DriveStudio agent start state matrix
    x = mat[0][3]
    y = mat[1][3]
    z = mat[2][3]
    # To get yaw from rotation matrix, simple approach if no pitch/roll:
    yaw = math.degrees(math.atan2(mat[1][0], mat[0][0]))
    return carla.Transform(carla.Location(x=x, y=y, z=z+0.5), carla.Rotation(yaw=yaw))

def main():
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(10.0)
    
    # Try to load the OpenDrive map if needed, or let ScenarioRunner handle it
    print("Connecting to traffic manager...")
    tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(False) # Wait till ego spawns? Might need to coordinate with B2DVL
    
    world = client.get_world()
    
    with open('cc8c0bf57f984915a77078b10eb33198/agents.json', 'r') as f:
        agents_data = json.load(f)['agents']
        
    blueprints = world.get_blueprint_library()
    
    spawned_actors = []
    for agent in agents_data:
        bp_name = agent.get('blueprint')
        if not bp_name:
            continue
        try:
            bp = blueprints.find(bp_name)
            if bp.has_attribute('color'):
                color = f"{agent['color'][0]},{agent['color'][1]},{agent['color'][2]}"
                bp.set_attribute('color', color)
            
            transform = get_transform(agent['first_transform'])
            actor = world.try_spawn_actor(bp, transform)
            if actor:
                print(f"Spawned {bp_name} at {transform.location}")
                spawned_actors.append(actor)
                if 'vehicle' in bp_name:
                    actor.set_autopilot(True, tm.get_port())
                    tm.ignore_lights_percentage(actor, random.randint(0, 100)) # Optional tm config
        except Exception as e:
            print(f"Failed to spawn {bp_name}: {e}")
            
    print(f"Spawned {len(spawned_actors)} actors in total. Traffic manager is handling them.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Destroying actors")
        for a in spawned_actors:
            a.destroy()

if __name__ == '__main__':
    main()
