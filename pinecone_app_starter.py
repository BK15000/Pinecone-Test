import pandas as pd
import pinecone
from dotenv import load_dotenv
import os
load_dotenv()


PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
index_name = os.getenv('index_name')

pc = pinecone.Pinecone(PINECONE_API_KEY)

index = pc.Index(index_name)

def print_query_results(results, query_description=""):
    """
    Print formatted query results from Pinecone index search
    Args:
        results: Dictionary containing Pinecone query results
        query_description: Optional string describing the query
    """
    print(f"\nSearch Results{' for ' + query_description if query_description else ''}")
    print(f"Found {len(results['matches'])} matches")
    for match in results['matches']:
        print(f"Score: {match['score']:.4f}")
        print(f"Title: {match['metadata']['Title']}")
        print(f"Review Summary: {match['metadata']['review/summary']}")
        print(f"Rating: {match['metadata']['review/score']}")
        print("-" * 50)


def get_query_embedding(query_text):
    return pc.inference.embed(
        model="multilingual-e5-large",
        inputs=[query_text],
        parameters={"input_type": "query", "truncate": "END"}
    )[0]['values']
    


# QUERY TYPE 1: Filtering/Projecting
print("\n--- QUERY TYPE 1: Search for ID ---")

try:
    filtered_results = index.query(
         #Type here
    )

    print_query_results(filtered_results, "filtered search for high-rated books")
except Exception as e:
    print(f"Error during filtered search: {e}")



# QUERY TYPE 2: Filtering/Projecting 
print("\n--- QUERY TYPE 2: Metatda Filtering ---")
query_text = "book" # Just a placeholder vector as all instances are books
query_embedding = get_query_embedding(query_text)

try:
    # Pure metadata filtering without considering vector similarity
    filtered_results = index.query(
       #Type here
    )

    print_query_results(filtered_results, "filtered search for high-rated books")
except Exception as e:
    print(f"Error during filtered search: {e}")




# QUERY TYPE 3: Semantic Search 
print("\n--- QUERY TYPE 3: Semantic Search ---")
semantic_query = "" #Type here
semantic_embedding = get_query_embedding(semantic_query)

try:
    semantic_results = index.query(
         #Type here
    )
    
    print_query_results(semantic_results, "semantic search for sci-fi books")
except Exception as e:
    print(f"Error during semantic search: {e}")





# QUERY TYPE 4: Hybrid Search 
print("\n--- QUERY TYPE 4: Hybrid Search ---")
hybrid_query = "" #Type here
hybrid_embedding = get_query_embedding(hybrid_query)

try:
    hybrid_results = index.query(
        #Type here
    )
    
    print_query_results(hybrid_results, "hybrid search for high-rated romance books")
except Exception as e:
    print(f"Error during hybrid search: {e}")