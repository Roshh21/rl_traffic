import xml.etree.ElementTree as ET

# Load your .net.xml file
input_file = "map_tls_fixed.net.xml"
output_file = "map_tls_fixed_cleaned.net.xml"

# Parse XML
tree = ET.parse(input_file)
root = tree.getroot()

# Optionally: Fix indentation or reformat the output
def pretty_print(element, indent='  ', level=0):
    spaces = '\n' + level*indent
    if len(element):
        if not element.text or not element.text.strip():
            element.text = spaces + indent
        for child in element:
            pretty_print(child, indent, level+1)
        if not child.tail or not child.tail.strip():
            child.tail = spaces
    if level and (not element.tail or not element.tail.strip()):
        element.tail = spaces

pretty_print(root)  # Indent for readability

# Example: Print types for validation
print("\nNetwork link types:")
for net_type in root.findall('type'):
    print(f"Type id: {net_type.get('id')}, priority: {net_type.get('priority')}")

# Save the cleaned file
tree.write(output_file, encoding='UTF-8', xml_declaration=True)
print(f"✓ Cleaned network saved to {output_file}")
