#!/usr/bin/env python3
"""
Wrapper script to convert OBJ to GLB using system Blender
This replaces the bpy-dependent convert_obj_to_glb function
"""

import subprocess
import sys
import os
import tempfile

def convert_obj_to_glb_external(obj_path, glb_path):
    """Convert OBJ to GLB using external Blender process"""
    
    # Create a temporary Python script for Blender
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(f'''
import bpy
import sys

# Clear existing objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import OBJ
bpy.ops.wm.obj_import(filepath="{obj_path}")

# Select all mesh objects
bpy.ops.object.select_all(action='DESELECT')
for obj in bpy.context.scene.objects:
    if obj.type == 'MESH':
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj

# Apply smooth shading
bpy.ops.object.shade_smooth()

# Export as GLB
bpy.ops.export_scene.gltf(
    filepath="{glb_path}",
    use_selection=True,
    export_format='GLB'
)

sys.exit(0)
''')
        script_path = f.name
    
    # Find Blender executable
    blender_paths = [
        '/Applications/Blender.app/Contents/MacOS/Blender',
        '/usr/local/bin/blender',
        '/opt/homebrew/bin/blender',
    ]
    
    blender_exe = None
    for path in blender_paths:
        if os.path.exists(path):
            blender_exe = path
            break
    
    if not blender_exe:
        # Try to find in PATH
        try:
            result = subprocess.run(['which', 'blender'], capture_output=True, text=True)
            if result.returncode == 0:
                blender_exe = result.stdout.strip()
        except:
            pass
    
    if not blender_exe:
        print("Error: Blender not found. Please install Blender.")
        return False
    
    # Run Blender in background mode
    try:
        cmd = [blender_exe, '--background', '--python', script_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Clean up temp script
        os.unlink(script_path)
        
        if result.returncode == 0:
            print(f"Successfully converted {obj_path} to {glb_path}")
            return True
        else:
            print(f"Blender conversion failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"Error running Blender: {e}")
        if os.path.exists(script_path):
            os.unlink(script_path)
        return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python blender_convert.py input.obj output.glb")
        sys.exit(1)
    
    convert_obj_to_glb_external(sys.argv[1], sys.argv[2])
