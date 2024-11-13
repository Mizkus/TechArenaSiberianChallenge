import subprocess
import numpy as np
import struct
import os
import sys
import tqdm

def read_ground_truth(file_path, query_index):
    with open(file_path, 'rb') as file:
        dim = struct.unpack('i', file.read(4))[0]  
        file.seek(query_index * (dim + 1) * 4)  
        buffer = file.read(4 * (100 + 1))  
        
        if not buffer: 
            return None
        
        indices = struct.unpack('i' * 101, buffer)
        return indices[1:11]  

def read_query_vector(file_path, index):
    with open(file_path, 'rb') as file:
        dim = struct.unpack('i', file.read(4))[0]  
        file.seek(index * (dim + 1) * 4)  
        buffer = file.read(4 * (dim + 1))  
        if not buffer:
            return None
        query_vector = struct.unpack('f' * dim, buffer[4:])  
        return np.array(query_vector)
    
def count_vectors(file_path):
    with open(file_path, 'rb') as file:
        dim = struct.unpack('i', file.read(4))[0] 
        count = 0
        
        while True:
            buffer = file.read(4 * (dim + 1)) 
            if not buffer:
                break  
            
            count += 1
    
    return count

def run_benchmark(data_path, build_index, scan_index):
    dataset_path = os.path.join(data_path, 'sift_base.fvecs')
    subprocess.run(['python', build_index, dataset_path])

    total_correct = 0

    query_count = count_vectors(os.path.join(data_path, 'sift_query.fvecs'))

    for i in tqdm.tqdm(range(query_count)):
        query_vector = read_query_vector(os.path.join(data_path, 'sift_query.fvecs'), i)
        if query_vector is None:
            break

        query_str = ','.join(map(str, query_vector))
        
        result = subprocess.run(['python', scan_index, f'[{query_str}]'], capture_output=True, text=True)
        predicted_indices = list(map(int, result.stdout.strip().split(',')))

        true_indices = read_ground_truth(os.path.join(data_path, 'sift_groundtruth.ivecs'), i)
        if true_indices is None:
            break
        
        correct_count = sum(1 for idx in predicted_indices if idx in true_indices)

        total_correct += correct_count
        recall = total_correct / ((i + 1) * 10)  

        print(f'Correct count: {correct_count} / 10')
        print(f'Recall: {recall}')



if __name__ == "__main__":    
    dataset_path = "../sift/"
    build_index = 'build_index.py'
    scan_index = 'scan_index.py'
    run_benchmark(dataset_path, build_index, scan_index)
