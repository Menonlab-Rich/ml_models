import numpy as np
import os

def check_npz_files(directory):
    ones = 0
    twos = 0
    
    for filename in os.listdir(directory):
        if filename.endswith('.npz'):
            filepath = os.path.join(directory, filename)
            data = np.load(filepath)
            mask = data['mask']
            
            ones += (mask == 1).any()
            twos += (mask == 2).any()
    
    print(f'ones: {ones}')
    print(f'twos: {twos}')

# Example usage
if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description="validation script")
    parser.add_argument("path")
    args = parser.parse_args()
    check_npz_files(args.path)
