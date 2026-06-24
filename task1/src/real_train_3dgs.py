import os
import subprocess
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Real Wrapper for 3DGS Training (Object A / Background)")
    parser.add_argument("--source", type=str, required=True, help="Path to COLMAP images/database")
    parser.add_argument("--model_path", type=str, required=True, help="Output directory")
    parser.add_argument("--iterations", type=int, default=30000)
    args = parser.parse_args()
    
    # Check if 3dgs repo exists
    repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "submodules", "gaussian-splatting"))
    if not os.path.exists(repo_dir):
        print(f"Error: 3DGS repository not found at {repo_dir}")
        sys.exit(1)
        
    train_script = os.path.join(repo_dir, "train.py")
    
    # Construct the actual training command
    cmd = [
        sys.executable, train_script,
        "-s", args.source,
        "-m", args.model_path,
        "--iterations", str(args.iterations)
    ]
    
    print("="*50)
    print("Running actual 3D Gaussian Splatting Training...")
    print("Command:", " ".join(cmd))
    print("="*50)
    
    # Execute the command
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error code {e.returncode}")
        print("Note: If it failed due to 'diff-gaussian-rasterization' not found, you must install the submodules first:")
        print("      pip install submodules/gaussian-splatting/submodules/diff-gaussian-rasterization")
        print("      pip install submodules/gaussian-splatting/submodules/simple-knn")
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()
