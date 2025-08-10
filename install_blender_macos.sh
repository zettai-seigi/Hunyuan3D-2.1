#!/bin/bash

echo "🔧 Blender Python (bpy) Setup for macOS"
echo "========================================"
echo ""

# Check if Blender is installed
if command -v blender &> /dev/null; then
    echo "✅ Blender found in PATH: $(which blender)"
    BLENDER_PATH=$(which blender)
elif [ -d "/Applications/Blender.app" ]; then
    echo "✅ Blender.app found in Applications"
    BLENDER_PATH="/Applications/Blender.app/Contents/MacOS/Blender"
else
    echo "❌ Blender not found. Please install Blender first."
    echo ""
    echo "Options to install Blender:"
    echo "1. Download from https://www.blender.org/download/"
    echo "2. Install via Homebrew: brew install --cask blender"
    echo ""
    exit 1
fi

echo ""
echo "📝 Blender Python (bpy) cannot be installed directly via pip."
echo "However, we can create a wrapper script to use Blender's Python."
echo ""

# Create a wrapper script for using Blender's Python
cat > blender_convert.py << 'EOF'
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
EOF

echo "✅ Created blender_convert.py wrapper script"
echo ""

# Create an alternative mesh_utils patch
cat > mesh_utils_blender_patch.py << 'EOF'
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
EOF

echo "✅ Created mesh_utils_blender_patch.py"
echo ""
echo "📌 Setup complete!"
echo ""
echo "The app will now use external Blender for OBJ to GLB conversion when bpy is not available."
echo ""
echo "Note: For best performance, consider using alternative GLB export methods"
echo "that don't require Blender, such as using trimesh or pygltflib directly."