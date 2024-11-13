import numpy as np
import struct
import os
import sys
import faiss

def fvecs_read_batch(file, dim, start, batch_size):
    file.seek(start * (dim + 1) * 4)  

    buffer = file.read(batch_size * (dim + 1) * 4)
    if not buffer:
        raise IOError("Error reading batch from file.")

    vectors = []
    for i in range(batch_size):
        vector_dim = struct.unpack('i', buffer[i * (dim + 1) * 4: i * (dim + 1) * 4 + 4])[0]
        if vector_dim != dim:
            raise ValueError("Non-uniform vector sizes in file.")
        vector = struct.unpack('f' * dim, buffer[i * (dim + 1) * 4 + 4: (i + 1) * (dim + 1) * 4])
        vectors.append(vector)

    return np.array(vectors)

def build_index(filename, m=16, ef_construction=80):
    if not os.path.exists(filename):
        print(f"Error: Unable to open file: {filename}")
        return

    with open(filename, 'rb') as file:
        file.seek(0, os.SEEK_END)
        size = file.tell()
        file.seek(0, os.SEEK_SET)

        if size == 0:
            print("Dataset is empty.")
            return

        dim = struct.unpack('i', file.read(4))[0]
        if dim <= 0:
            print(f"Invalid dimension in file: {filename}")
            return

        num_vectors = size // ((dim + 1) * 4)
        batch_size = 100000

        index = faiss.IndexHNSWSQ(dim, faiss.ScalarQuantizer.QT_8bit, m)
        train_vectors = fvecs_read_batch(file, dim, 0, min(batch_size, num_vectors))
        index.train(train_vectors)
        index.hnsw.efConstruction = ef_construction  
        

        for start in range(0, num_vectors, batch_size):
            current_batch_size = min(batch_size, num_vectors - start)
            batch_vectors = fvecs_read_batch(file, dim, start, current_batch_size)
            index.add(batch_vectors)
 
        faiss.write_index(index, "database.index")
            
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} dataset_name")
        sys.exit(1)
    dataset = sys.argv[1]
    build_index(dataset)
    print("Index successfully built")
