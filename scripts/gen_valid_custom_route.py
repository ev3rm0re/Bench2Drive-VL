#!/usr/bin/env python3
import argparse
import math
import xml.etree.ElementTree as ET

import carla
import networkx as nx
from agents.navigation.global_route_planner import GlobalRoutePlanner


def loc_distance(a: carla.Location, b: carla.Location) -> float:
    dx = a.x - b.x
    dy = a.y - b.y
    dz = a.z - b.z
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def to_position(parent, t: carla.Transform, z_offset: float = 0.5):
    ET.SubElement(
        parent,
        "position",
        {
            "x": f"{t.location.x:.6f}",
            "y": f"{t.location.y:.6f}",
            "z": f"{t.location.z + z_offset:.6f}",
            "pitch": f"{t.rotation.pitch:.6f}",
            "yaw": f"{t.rotation.yaw:.6f}",
            "roll": f"{t.rotation.roll:.6f}",
        },
    )


def to_trigger(parent, t: carla.Transform, z_offset: float = 0.5):
    ET.SubElement(
        parent,
        "trigger_point",
        {
            "x": f"{t.location.x:.6f}",
            "y": f"{t.location.y:.6f}",
            "z": f"{t.location.z + z_offset:.6f}",
            "pitch": f"{t.rotation.pitch:.6f}",
            "yaw": f"{t.rotation.yaw:.6f}",
            "roll": f"{t.rotation.roll:.6f}",
        },
    )


def build_route_xml(out_xml: str, town: str, start_t: carla.Transform, end_t: carla.Transform):
    routes = ET.Element("routes")
    route = ET.SubElement(routes, "route", {"id": "0", "town": town})
    ET.SubElement(
        route,
        "weather",
        {
            "cloudiness": "0.0",
            "precipitation": "0.0",
            "precipitation_deposits": "0.0",
            "wind_intensity": "0.0",
            "sun_azimuth_angle": "0.0",
            "sun_altitude_angle": "90.0",
            "fog_density": "0.0",
            "wetness": "0.0",
            "dropness": "0.0",
        },
    )

    waypoints = ET.SubElement(route, "waypoints")
    to_position(waypoints, start_t)
    to_position(waypoints, end_t)

    scenarios = ET.SubElement(route, "scenarios")
    scenario = ET.SubElement(
        scenarios,
        "scenario",
        {"name": "CustomScenario", "type": "CustomDriveStudioScenario"},
    )
    to_trigger(scenario, start_t)

    tree = ET.ElementTree(routes)
    ET.indent(tree, space="  ")
    tree.write(out_xml, encoding="utf-8", xml_declaration=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", default="singapore-onenorth")
    parser.add_argument("--out", default="custom_route.xml")
    parser.add_argument("--min-distance", type=float, default=80.0)
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)
    world = client.get_world()
    cmap = world.get_map()
    grp = GlobalRoutePlanner(cmap, 2.0)

    spawn_points = cmap.get_spawn_points()
    if len(spawn_points) < 2:
        raise RuntimeError("Not enough spawn points in current map")

    best = None
    for i, s in enumerate(spawn_points):
        s_node = grp._localize(s.location)
        if not s_node:
            continue

        for j, e in enumerate(spawn_points):
            if i == j:
                continue
            if loc_distance(s.location, e.location) < args.min_distance:
                continue

            e_node = grp._localize(e.location)
            if not e_node:
                continue

            connected = False
            for u in s_node:
                for v in e_node:
                    if u != v and nx.has_path(grp._graph, u, v):
                        connected = True
                        break
                if connected:
                    break

            if connected:
                best = (s, e)
                break
        if best:
            break

    if not best:
        raise RuntimeError("No connected spawn-point pair found")

    start_t, end_t = best
    print("Selected start:", start_t)
    print("Selected end:", end_t)
    build_route_xml(args.out, args.town, start_t, end_t)
    print(f"Wrote route to {args.out}")


if __name__ == "__main__":
    main()
