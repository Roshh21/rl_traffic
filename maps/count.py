import xml.etree.ElementTree as ET

route_file = "map.rou_rl_ready.xml"  # Change to your route file name

tree = ET.parse(route_file)
root = tree.getroot()

vehicle_count = len(root.findall('vehicle'))
print(f"Number of vehicles remaining: {vehicle_count}")
