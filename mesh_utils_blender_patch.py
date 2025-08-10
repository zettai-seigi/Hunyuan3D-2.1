"""
Patch for mesh_utils.py to use external Blender when bpy is not available
"""

import os
import subprocess
import sys

def convert_obj_to_glb_fallback(obj_path, glb_path, **kwargs):
    """Fallback function using external Blender process"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    convert_script = os.path.join(script_dir, '..', '..', 'blender_convert.py')
    
    if not os.path.exists(convert_script):
        print("Warning: blender_convert.py not found. Cannot convert to GLB.")
        return False
    
    try:
        result = subprocess.run(
            [sys.executable, convert_script, obj_path, glb_path],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except Exception as e:
        print(f"Error calling external Blender: {e}")
        return False

print("Blender fallback patch loaded")
