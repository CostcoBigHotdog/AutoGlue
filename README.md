# AutoGlue

An implementation for automatic BDD step-definition (glue code) generation using a retrieval-augmented few-shot prompting framework.


## 1. Environment

- Python (recommended 3.10)
- Required packages are listed in `requirements.txt`

```
pip install -r requirements.txt
```


## 2. Create `.env` File

Create a `.env` file in the project root:

```
OPENROUTER_API_KEY=your_key_here
MODEL_NAME=your_model_name
```

By default, the system uses the OpenRouter API.  
If you use another provider, adjust the model and API base settings in the code accordingly.


## 3. Run Evaluation

### Evaluate all datasets (default: 3-shot prompting)

```
python evaluate.py --n_shots 3
```

### Evaluate only the first N steps for each BDD project

```
python evaluate.py --n_shots 3 --num_steps 50
```


## 4. View Results

All outputs and scores are stored in the `logs/` directory.  

```
logs/glue_code_generate_3shot.log
```
