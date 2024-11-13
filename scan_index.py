import numpy as np
import struct
import os
import sys
import faiss  

def clean_query_input(query):
    return query.replace('[', '').replace(']', '')

def main(query):
    query = clean_query_input(query)

    if not os.path.exists('database.index'):
        print("Error: Index file not found.")
        return

    index = faiss.read_index('database.index') 

    index.hnsw.efSearch = 2048

    query_vector = np.array([float(x) for x in query.split(',')]).astype('float32')
    query_vector = query_vector.reshape(1, -1)

    if query_vector.shape[1] != index.d:
        print(f"Error: Query vector dimension {query_vector.shape[1]} does not match index dimension {index.d}.")
        return

    distances, indices = index.search(query_vector, 10)

    result = ','.join(str(idx) for idx in indices[0])
    print(result)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <query vector>")
        sys.exit(1)
    query = sys.argv[1]
    main(query)
