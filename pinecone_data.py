import pandas as pd
import pinecone
import time
import hashlib
from dotenv import load_dotenv
import os
load_dotenv()

# Initialize Pinecone
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
index_name = os.getenv('index_name')
pc = pinecone.Pinecone(PINECONE_API_KEY)


# Check if index exists
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1024,
        metric="cosine",
        spec=pinecone.ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ) 
    )
    print(f"Created new index: {index_name}")
    
    index = pc.Index(index_name)
    df = pd.read_csv('books_rating.csv')
    df = df.fillna('')

    # Use a fixed subset - first 95 rows (not random)
    batch_size = 95
    df_sample = df.head(batch_size)

    documents = []
    for idx, row in df_sample.iterrows():
        # Combine review text and summary for better context
        text_to_embed = f"{row['review/summary']} {row['review/text']}"
        
        # Create a deterministic ID based on the row content
        # This ensures the same ID is generated for the same data
        id_string = f"{row.get('Id', '')}-{row.get('User_id', '')}-{row.get('review/time', '')}"
        deterministic_id = hashlib.md5(id_string.encode()).hexdigest()
        
        # Create a document with all metadata
        doc = {
            "id": deterministic_id,  
            "text": text_to_embed,
            "metadata": {
                "Id": row.get('Id', ''),
                "Title": row.get('Title', ''),
                "Price": row.get('Price', ''),
                "User_id": row.get('User_id', ''),
                "profileName": row.get('profileName', ''),
                "review/helpfulness": row.get('review/helpfulness', ''),
                "review/score": float(row.get('review/score', 0)),
                "review/time": int(row.get('review/time', 0)),
                "review/summary": row.get('review/summary', ''),
            }
        }
        documents.append(doc)

    print(f"Prepared {len(documents)} documents for embedding")

    # Create embeddings for the batch
    print("Creating embeddings for batch")
    batch_embeddings = pc.inference.embed(
        model="multilingual-e5-large",
        inputs=[d['text'] for d in documents],
        parameters={"input_type": "passage", "truncate": "END"}
    )

    # Prepare vectors for upsert
    vectors = []
    for d, e in zip(documents, batch_embeddings):
        vectors.append({
            "id": d['id'],
            "values": e['values'],
            "metadata": d['metadata']
        })

    # Upsert the batch of vectors
    print(f"Upserting batch of {len(vectors)} vectors")
    try:
        index.upsert(
            vectors=vectors,
            namespace="books"
        )
        print(f"Successfully upserted {len(vectors)} vectors")
    except Exception as e:
        print(f"Error upserting batch: {e}")
else:
    print(f"Index already exists: {index_name}")




def wait_for_index(index, expected_count, timeout=30, interval=2):
    start = time.time()
    while time.time() - start < timeout:
        stats = index.describe_index_stats()
        count = stats['total_vector_count']
        print(f"Waiting for vectors... {count}/{expected_count} available")
        if count >= expected_count:
            return True
        time.sleep(interval)
    print("Timeout: Not all vectors are indexed yet.")
    return False

wait_for_index(index, 95) 