import traci
import xml.etree.ElementTree as ET

# Configuration
sumo_cfg = "maps/sumo.sumocfg"
route_file = "maps/map.rou_rl_ready.xml"
output_route_file = "maps/cleaned_routes.rou.xml"

# Start SUMO
sumo_cmd = [
    'sumo',  # Use 'sumo-gui' if you want to visualize
    '-c', sumo_cfg,
    '--time-to-teleport', '300',  # Allow teleport after 300s jam
    '--no-step-log', 'true',
    '--no-warnings', 'false',  # Keep warnings to detect teleports
]

traci.start(sumo_cmd)

teleported_ids = set()

# Run simulation and track teleports
print("Running SUMO simulation to detect teleports...")
step = 0
while traci.simulation.getMinExpectedNumber() > 0:
    traci.simulationStep()
    step += 1
    
    # Get all vehicles that have teleported this step
    # TraCI doesn't have a direct "getTeleported" call, but we can check for vehicles
    # that suddenly disappear and reappear or check warnings
    
    # Alternative: Check for vehicles removed due to teleport
    # SUMO logs teleport events; we'll parse those from the console or use a workaround
    
    # For now, we'll use a proxy: vehicles with very low speed stuck for long time
    # A better approach is to monitor SUMO output logs for "Teleporting vehicle" messages
    
    if step % 100 == 0:
        print(f"Step {step}: {traci.simulation.getMinExpectedNumber()} vehicles remaining")

traci.close()
print("Simulation complete.")

# Note: To properly detect teleports, you should:
# 1. Run SUMO with output redirected to capture warnings
# 2. Parse "Teleporting vehicle 'ID'" messages from output
# 3. Or use SUMO's message subscriptions

# For now, let's assume you manually collected teleported IDs or parsed logs
# Example: teleported_ids = {'veh_0042', 'veh_0153', 'veh_0891'}

# If you want to automate this, run SUMO with stderr capture:
import subprocess
import re

print("\nRe-running SUMO to capture teleport warnings...")
result = subprocess.run(
    sumo_cmd,
    stderr=subprocess.PIPE,
    stdout=subprocess.PIPE,
    text=True
)

# Parse teleport warnings from stderr
teleport_pattern = r"Teleporting vehicle '([^']+)'"
teleported_ids = set(re.findall(teleport_pattern, result.stderr))

print(f"Found {len(teleported_ids)} teleported vehicles: {teleported_ids}")

# Remove teleported vehicles from route file
if len(teleported_ids) > 0:
    tree = ET.parse(route_file)
    root = tree.getroot()
    
    vehicles_removed = 0
    for vehicle in root.findall('vehicle'):
        if vehicle.get('id') in teleported_ids:
            root.remove(vehicle)
            vehicles_removed += 1
    
    tree.write(output_route_file, encoding='UTF-8', xml_declaration=True)
    print(f"✓ Removed {vehicles_removed} teleported vehicles")
    print(f"✓ Cleaned route file saved as: {output_route_file}")
else:
    print("No teleports detected, no changes made.")