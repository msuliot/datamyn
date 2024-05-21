# DATAMYN - Blades of Grass - AI Assistant
## Blades of Grass is an AI assistant application that leverages OpenAI, Pinecone, MongoDB, and Gradio to answer user queries based on a specified context. It integrates with these services to retrieve relevant information and provide accurate and context-aware responses.

Key Features
- Dynamic Query Handling: Processes user queries to generate relevant responses using OpenAI's language model.
- Contextual Responses: Retrieves relevant chunks of text from Pinecone and MongoDB to create contextually accurate answers.
- Interactive UI: Uses Gradio to provide an interactive user interface for querying the AI assistant.
- Configurable Prompts: Allows customization of system and user prompts for tailored interactions.

## Git Repositories
- https://github.com/msuliot/texten.git
- https://github.com/msuliot/webtexten.git
- https://github.com/msuliot/chunken.git
- https://github.com/msuliot/datamyn.git

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)

## Prerequisites

Before you begin, ensure you have met the following requirements:
- You have installed Python 3.7 or later.
- You have a working internet connection.

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/msuliot/datamyn.git
    cd texten
    ```

2. **Set up a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the DATAMYN application, use the following command:

```bash
python app.py
```
or
```bash
python command.py "query"
```
