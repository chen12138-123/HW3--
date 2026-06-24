import os
import subprocess
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Real Wrapper for Threestudio Text-to-3D (Object B)")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for generation")
    parser.add_argument("--out", type=str, required=True, help="Output directory")
    args = parser.parse_args()
    
    repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "submodules", "threestudio"))
    if not os.path.exists(repo_dir):
        print(f"Error: Threestudio repository not found at {repo_dir}")
        sys.exit(1)
        
    launch_script = os.path.join(repo_dir, "launch.py")
    config_file = os.path.join(repo_dir, "configs", "magic3d-refine-sd.yaml") # Using magic3d as representative SDS
    
    cmd = [
        sys.executable, launch_script,
        "--config", config_file,
        "--train",
        "--gpu", "0",
        f"system.prompt_processor.prompt='{args.prompt}'",
        f"resume={args.out}" # Using output dir to save results
    ]
    
    print("="*50)
    print("Running actual Threestudio Text-to-3D (SDS) Training...")
    print("Command:", " ".join(cmd))
    print("="*50)
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error code {e.returncode}")
        print("Note: Ensure all threestudio dependencies (ninja, xformers, tinycudann) are installed.")
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()
