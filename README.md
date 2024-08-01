# climate_pal
Climate PAL - Climate Analysis and Projection Language Model

## Setup

Place the following contents into `langchain_demo/key.py`:

```python
import os

def init():
    os.environ["OPENAI_API_KEY"] = '' # Put your key here!
```

### Langflow environment
1. ```python -m pip install langflow -U```
2. ```python -m langflow run```
3. import the file ```Climate Prediction Updated.json``` via the Langflow GUI


### Langchain Demo

Sonia's version:
```python
conda create --name giss2 python==3.11.4
conda activate giss2
conda install numpy pandas openai scikit-learn jupyter
pip install langchain langchain_core langchain_openai langchain_experimental openpyxl
```

``` conda create -n langchain_env python=3.11.7```<!-- I added this assuming you want to install the packages in the conda environment -->    
``` conda activate langchain_env```    
```conda install openai```  
```pip install chromadb gradio```  
```conda install langchain -c conda-forge```  
```pip install langchain-openai langchain-community lark```

## Important files

```langchain_demo_reformat.py``` declates the Langchain Documents right in the code, as opposed to reading them in from JSON files. I wanted to experiment with making ```page_content``` be the comment description and so far this seems to be a better approach. 

```gradio_app.py``` contains pretty much the same code as the above file, but we load in data through a JSON loader

