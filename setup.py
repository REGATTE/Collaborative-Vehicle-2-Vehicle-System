import os
import subprocess
import shutil
from pathlib import Path

def setup_project():
    """
    Sets up the project by:
    1. Installing required packages from requirements.txt
    2. Replacing the mask.py file in carla_bird_eye_view library
    """
    # Create requirements.txt with necessary packages
    requirements = [
        "carla-birdeye-view",
        # Add other required packages here
    ]
    
    with open("requirements.txt", "w") as f:
        for req in requirements:
            f.write(f"{req}\n")
    
    # Install requirements
    print("Installing required packages...")
    subprocess.check_call(["pip", "install", "-r", "requirements.txt"])

    # Create birdview cache directory structure
    cache_path = Path("birdview_v2_cache/Carla/Maps")
    cache_path.mkdir(parents=True, exist_ok=True)
    print(f"Created directory structure: {cache_path}")
    
    # Find the carla_bird_eye_view installation directory
    try:
        import carla_birdeye_view
        library_path = Path(carla_birdeye_view.__file__).parent
        mask_path = library_path / "mask.py"
        
        # Backup original mask.py
        if mask_path.exists():
            backup_path = mask_path.with_name("mask.py.backup")
            shutil.copy2(mask_path, backup_path)
            print(f"Original mask.py backed up to {backup_path}")
        
        # Replace with your custom mask.py
        custom_mask_path = Path("AdditionalThirdParty/mask.py")  # Update this path
        if custom_mask_path.exists():
            shutil.copy2(custom_mask_path, mask_path)
            print(f"Successfully replaced mask.py with your custom version")
        else:
            print(f"Error: Custom mask.py not found at {custom_mask_path}")
            
    except ImportError:
        print("Error: carla_birdeye_view package not found. Please check installation.")
    except Exception as e:
        print(f"Error during setup: {str(e)}")

if __name__ == "__main__":
    setup_project()