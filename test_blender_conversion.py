#!/usr/bin/env python3
"""
Test script to verify Blender OBJ to GLB conversion works
"""

import sys
import os
sys.path.insert(0, './hy3dpaint')

from DifferentiableRenderer.mesh_utils import convert_obj_to_glb

# Create a simple test OBJ file
test_obj_content = """# Simple triangle mesh
v 0 0 0
v 1 0 0
v 0 1 0
f 1 2 3
"""

# Write test OBJ file
with open('test_triangle.obj', 'w') as f:
    f.write(test_obj_content)

print("Testing OBJ to GLB conversion with external Blender...")
success = convert_obj_to_glb('test_triangle.obj', 'test_triangle.glb')

if success and os.path.exists('test_triangle.glb'):
    print("✅ SUCCESS! Blender conversion works.")
    print("   Advanced features are now enabled.")
    # Clean up test files
    os.remove('test_triangle.obj')
    os.remove('test_triangle.glb')
else:
    print("❌ FAILED. Check Blender installation.")