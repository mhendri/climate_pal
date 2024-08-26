# climate_pal
Climate PAL - Climate Analysis and Projection Language Model

## Setup
1. In the `langchain_demo` directory create a folder called `data`.
2. Create and place the following content into `langchain_demo/key.py` (Note: you need to create `key.py`):

    ```python
    import os

    def init():
        os.environ["OPENAI_API_KEY"] = '' # Put your key here!
    ```
3. clone the repository at [cmip6-cmor-tables](https://github.com/PCMDI/cmip6-cmor-tables) and place it at the same level directory as `climate_pal`.


### Langflow environment
Note: does not work on Safari browser

1. ```python -m pip install langflow -U```
2. ```python -m langflow run```
3. import the file ```Climate Prediction Updated.json``` via the Langflow GUI


### Langchain Demo

```conda create -n langchain_env python=3.11.7```

```conda activate langchain_env``` 

```pip install -r requirements.txt ```   

In the `langchain_demo` directory run:

`streamlit run streamlit_assistant.py`
