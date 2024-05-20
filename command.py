import os
from dotenv import load_dotenv, find_dotenv
import argparse

import msuliot.openai_helper as oai # https://github.com/msuliot/package.helpers.git
from msuliot.mongo_helper import MongoDatabase # https://github.com/msuliot/package.helpers.git
from msuliot.pinecone_helper import Pinecone # https://github.com/msuliot/package.helpers.git

import json

from env_config import envs
env = envs()


def create_system_prompt():
    system_prompt = f"""
    I'm an AI crafted with honesty, professionalism, empathy, and positivity, ready to assist based on the content you provided and my training. If I don't know something, I'll be upfront about it.  Respond as if I am a customer asking the question.
    """
    return system_prompt

 # create prompt for openai
def create_prompt(query, res):
    prompt_start = ("Answer the question based on the context of the question.\n\n" + "Context:\n") # also, do not discuss any Personally Identifiable Information.
    prompt_end = (f"\n\nQuestion: {query}\nAnswer:")
    prompt = (prompt_start + "\n\n---\n\n".join(res) + prompt_end)
    return prompt

# main function
def main(query, namespace):
    os.system('clear')
    # print("Start: Main function")
    print(f"Query: {query}")
    print(f"Namespace: {namespace}\n")

    # Initialize models and services
    model_for_openai_embedding = "text-embedding-3-small"
    model_for_openai_chat = "gpt-4o"
    pc = Pinecone(api_key=env.pinecone_key)
    index = pc.Index("blades-of-grass")
    oaie = oai.openai_embeddings(env.openai_key, model_for_openai_embedding)
    embed = oaie.execute(query)

    # Query Pinecone Index
    response_pine = index.query(
        namespace=namespace,
        vector=embed.data[0].embedding, 
        top_k=10, 
        include_metadata=True, 
        include_values=False,
    )

    oaic = oai.openai_chat(env.openai_key, model_for_openai_chat)

    source_files = []
    context = []
    source_info = []

    # Process matches from Pinecone response
    for match in response_pine['matches']:
        chunk_id = match['id']
        source_file = match['metadata']['source']
        chunk_number = match['metadata']['chunk_number']
        score = match['score']

        source_files.append(f"{source_file} - {chunk_number} - {score}")

        # Retrieve chunk text from MongoDB
        with MongoDatabase(env.mongo_uri) as client:
            chunk_text = client.get_document_by_chunk_id("blades-of-grass", namespace, chunk_id)
            text = chunk_text[0]['data'][0]['text']
            context.append(text)
            
            # Store source file name and text content in source_info
            source_info.append({
                'source': source_file,
                'chunk_number': chunk_number,
                'score': score,
                'content': text
            })

    # Generate prompt for OpenAI Chat model
    prompt = create_prompt(query, context)

    oaic.add_message("system", create_system_prompt())
    oaic.add_message("user", prompt)
    oaic.execute_stream()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple command line RAG model. Make sure to use quotes around your query.")
    parser.add_argument('query', type=str, help='This is the query to be answered')
    parser.add_argument('--namespace', type=str, help='An optional parameter to specify the namespace for the Pinecone index. Default is "demo24"')
    args = parser.parse_args()
    print(args.namespace)

    if args.namespace:
        main(args.query, args.namespace)
    else:
        main(args.query, "demo24")