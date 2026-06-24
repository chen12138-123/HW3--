import os
import subprocess
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Real Wrapper for Zero123 Image-to-3D (Object C)")
    parser.add_argument("--image", type=str, required=True, help="Input single image")
    parser.add_argument("--out", type=str, required=True, help="Output directory")
    args = parser.parse_args()
    
    repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "submodules", "threestudio"))
    if not os.path.exists(repo_dir):
        print(f"Error: Threestudio repository not found at {repo_dir}")
        sys.exit(1)
        
    launch_script = os.path.join(repo_dir, "launch.py")
    config_file = os.path.join(repo_dir, "configs", "zero123.yaml") 
    
    cmd = [
        sys.executable, launch_script,
        "--config", config_file,
        "--train",
        "--gpu", "0",
        f"data.image_path={args.image}",
        f"resume={args.out}"
    ]
    
    print("="*50)
    print("Running actual Threestudio Zero123 Image-to-3D Training...")
    print("Command:", " ".join(cmd))
    print("="*50)
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error code {e.returncode}")
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()
