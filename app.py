import os
from dotenv import load_dotenv, find_dotenv
import gradio as gr

import msuliot.openai_helper as oai # https://github.com/msuliot/package.helpers.git
from msuliot.mongo_helper import MongoDatabase # https://github.com/msuliot/package.helpers.git
from msuliot.pinecone_helper import Pinecone # https://github.com/msuliot/package.helpers.git

import json

from env_config import envs
env = envs()


def create_system_prompt():
    system_prompt = f"""
    You are a Honest, professional and positive. If there's something you don't know then just say so.
    """
    return system_prompt

 # create prompt for openai
def create_prompt(query, res):
    prompt_start = ("Answer the question based on the context of the question.\n\n" + "Context:\n") # also, do not discuss any Personally Identifiable Information.
    prompt_end = (f"\n\nQuestion: {query}\nAnswer:")
    prompt = (prompt_start + "\n\n---\n\n".join(res) + prompt_end)
    return prompt

# main function
def main(namespace, system_prompt, query):
    print("Start: Main function")
    
    # Initialize models and services
    model_for_openai_embedding = "text-embedding-3-small"
    model_for_openai_chat = "gpt-4o"
    pc = Pinecone(api_key=env.pinecone_key)
    index = pc.Index("blades-of-grass-demo")
    oaie = oai.openai_embeddings(env.openai_key, model_for_openai_embedding)
    embed = oaie.execute(query)

    # Query Pinecone Index
    response_pine = index.query(
        namespace=namespace,
        vector=embed.data[0].embedding, 
        top_k=5, 
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
            chunk_text = client.get_document_by_chunk_id("blades-of-grass-demo", namespace, chunk_id)
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

    oaic.add_message("system", system_prompt)
    oaic.add_message("user", prompt)
    oaic.execute_stream()
    response = oaic.execute() 
    
    # print('-' * 80)

    final_output = response #+ "\n\n" + "Sources used to answer the question:\n" + "\n".join(source_files)
    # print(final_output)
    # print('-' * 80)

    # Return both final_output and source_info
    return final_output, json.dumps(source_info, indent=4)



############## Gradio UI ################
dropdown_namespaces = ["demo24", "demo74"]

gr.close_all()

with gr.Blocks() as demo:
    gr.Markdown("# Blades of Grass Demo")
    gr.Markdown("Blades of Grass is an AI Assistant that leverages OpenAI, Pinecone, MongoDB and your files to answer your questions.")
    
    with gr.Row():
        with gr.Column():
            namespace = gr.Dropdown(label="Choose a namespace", choices=dropdown_namespaces, value="demo24")
            system_prompt = gr.Textbox(label="System Prompt", lines=3, value="I'm an AI crafted with honesty, professionalism, empathy, and positivity, ready to assist based on the content you provided and my training. If I don't know something, I'll be upfront about it.  Respond as if I am a customer asking the question.")
            user_prompt = gr.Textbox(label="Hello, my name is Aiden, your AI Assistant. How can I help?", lines=8, placeholder="")
            submit_btn = gr.Button("Submit")

        # response = gr.Textbox(label="Response", lines=32)
        response = gr.Markdown(label="Response", value="")

    # source = gr.Textbox(label="Source content", lines=38)
    source = gr.components.JSON(label="Source")

    submit_btn.click(
        fn=main,
        inputs=[namespace, system_prompt, user_prompt],
        outputs=[response, source]
    )

demo.launch(server_name="localhost", server_port=8765)

