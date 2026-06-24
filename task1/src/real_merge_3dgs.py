import numpy as np
from plyfile import PlyData, PlyElement
import argparse

def merge_3dgs_plys(bg_path, obj_paths, transforms, out_path):
    print(f"Loading background {bg_path}...")
    bg_data = PlyData.read(bg_path)
    bg_elements = bg_data.elements[0].data
    
    all_elements = [bg_elements]
    
    for i, (obj_path, transform) in enumerate(zip(obj_paths, transforms)):
        print(f"Loading object {obj_path}...")
        obj_data = PlyData.read(obj_path)
        obj_elements = obj_data.elements[0].data
        
        # Apply transformation (Scale, Translation)
        scale = transform.get('scale', 1.0)
        trans = transform.get('trans', [0,0,0])
        
        obj_elements['x'] = obj_elements['x'] * scale + trans[0]
        obj_elements['y'] = obj_elements['y'] * scale + trans[1]
        obj_elements['z'] = obj_elements['z'] * scale + trans[2]
        
        # Scale also affects the covariance matrix (scale_0, scale_1, scale_2)
        # scale_i is stored in log space in 3DGS
        obj_elements['scale_0'] += np.log(scale)
        obj_elements['scale_1'] += np.log(scale)
        obj_elements['scale_2'] += np.log(scale)
        
        all_elements.append(obj_elements)
        
    print("Concatenating Gaussians...")
    merged_elements = np.concatenate(all_elements)
    
    el = PlyElement.describe(merged_elements, 'vertex')
    print(f"Saving merged scene to {out_path}...")
    PlyData([el], text=False).write(out_path)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bg", type=str, required=True)
    parser.add_argument("--obj_a", type=str, required=True)
    parser.add_argument("--obj_b", type=str, required=True)
    parser.add_argument("--obj_c", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()
    
    transforms = [
        {'scale': 1.5, 'trans': [2.0, 2.0, 0.0]},    # Obj A
        {'scale': 1.0, 'trans': [-3.0, 0.0, 0.0]},   # Obj B
        {'scale': 1.2, 'trans': [0.0, -3.0, 0.0]}    # Obj C
    ]
    
    merge_3dgs_plys(args.bg, [args.obj_a, args.obj_b, args.obj_c], transforms, args.out)
