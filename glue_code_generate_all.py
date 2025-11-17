import os
import json
import logging
from typing import List, Dict, Optional
import asyncio
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from rank_bm25 import BM25Okapi
from codebleu import calc_codebleu

import argparse


DATASET_DIR = "dataset/Java"
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")


llm = ChatOpenAI(
    model=MODEL_NAME,
    openai_api_key=OPENROUTER_API_KEY,
    openai_api_base="https://openrouter.ai/api/v1",
)
template = "{prompt}"
prompt_template = PromptTemplate(template=template, input_variables=["prompt"])
chain = RunnableSequence(prompt_template, llm)


def load_dataset(path: str) -> List[Document]:
    with open(path, "r", encoding="utf-8") as f:
        dataset: List[Dict] = json.load(f)
    docs = []
    for item in dataset:
        docs.append(
            Document(
                page_content=item["annotation"],
                metadata={
                    "glue_code": item["glue_code"],
                    "keyword": item["keyword"],
                    "id": item["step_definition_id"],
                },
            )
        )
    return docs

def build_bm25(docs: List[Document]) -> BM25Okapi:
    corpus = [doc.page_content for doc in docs]
    tokenized_corpus = [d.split() for d in corpus]
    return BM25Okapi(tokenized_corpus)

def retrieve_similar_steps(query: str, exclude_id: int, docs: List[Document], bm25: BM25Okapi, top_k: int = 3) -> List[Document]:
    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    results = []
    for i in ranked_indices:
        if docs[i].metadata["id"] != exclude_id:
            results.append(docs[i])
        if len(results) >= top_k:
            break
    return results

def build_few_shot_prompt(target_doc: Document, docs: List[Document], bm25: BM25Okapi, top_k: int = 3) -> str:
    examples = retrieve_similar_steps(target_doc.page_content, target_doc.metadata["id"], docs, bm25, top_k=top_k)
    prompt_parts = []
    for i, ex in enumerate(examples):
        prompt_parts.append(
            f"Example {i+1}:\nGherkin Step:\n{ex.page_content}\nGlue Code:\n{ex.metadata['glue_code']}\n"
        )
    few_shot_examples = "\n".join(prompt_parts)
    prompt = f"""
You are an expert in Behavior-Driven Development (BDD).
Your task is to generate the Java glue code for the given Gherkin step.

{few_shot_examples}

Now, generate glue code for this new step:
Gherkin Step:
{target_doc.page_content}

Glue Code:
"""
    return prompt

async def generate_glue_code(target_doc: Document, docs: List[Document], bm25: BM25Okapi, n_shots: int = 3) -> str:
    logging.info(f"Generating glue code for step ID {target_doc.metadata['id']}")
    prompt = build_few_shot_prompt(target_doc, docs, bm25, top_k=n_shots)
    result = await asyncio.to_thread(chain.invoke, {"prompt": prompt})

    if isinstance(result, AIMessage):
        code_str = result.content
    elif isinstance(result, str):
        code_str = result
    else:
        code_str = str(result)

    logging.info(f"Generated glue code: \n{code_str}")
    return code_str

async def evaluate_dataset(dataset_path: str, n_shots: int = 3, num_steps: Optional[int] = None):
    docs = load_dataset(dataset_path)
    bm25 = build_bm25(docs)
    steps_to_eval = docs if num_steps is None else docs[:num_steps]

    all_scores = []
    for doc in steps_to_eval:
        pred = await generate_glue_code(doc, docs, bm25, n_shots=n_shots)
        ref = doc.metadata["glue_code"]
        try:
            score = calc_codebleu([ref], [pred], lang="java", weights=(0.1,0.1,0.4,0.4), tokenizer=None)
        except Exception as e:
            logging.error(f"CodeBLEU error for step ID {doc.metadata['id']}: {e}")
            score = {"CodeBLEU": 0.0}
        all_scores.append(score)
        logging.info(f"Step ID {doc.metadata['id']} CodeBLEU: {score}")
        print(f"{os.path.basename(dataset_path)} - Step ID {doc.metadata['id']} CodeBLEU: {score}")

    avg_scores = {k: sum(d.get(k,0) for d in all_scores)/len(all_scores) for k in all_scores[0]}
    print(f"{os.path.basename(dataset_path)} Average CodeBLEU: {avg_scores}")
    return all_scores, avg_scores

async def evaluate_all_datasets(base_dir: str, n_shots: int = 3, num_steps: Optional[int] = None):
    all_dataset_scores = {}
    overall_scores = []

    tasks = []
    dataset_paths = []
    for root, dirs, files in os.walk(base_dir):
        if "step_definitions.json" in files:
            dataset_path = os.path.join(root, "step_definitions.json")
            dataset_paths.append(dataset_path)
            tasks.append(evaluate_dataset(dataset_path, n_shots=n_shots, num_steps=num_steps))

    results = await asyncio.gather(*tasks)

    for dataset_path, (scores, avg_scores) in zip(dataset_paths, results):
        dataset_name = os.path.basename(os.path.dirname(dataset_path))
        all_dataset_scores[dataset_name] = avg_scores
        overall_scores.extend(scores)
        logging.info(f"{dataset_name} Average CodeBLEU per metric: {avg_scores}")
        print(f"{dataset_name} Average CodeBLEU per metric: {avg_scores}")


    overall_avg = {k: sum(d.get(k, 0) for d in overall_scores)/len(overall_scores) for k in overall_scores[0]}
    logging.info(f"\nOverall Average CodeBLEU across all datasets per metric: {overall_avg}")
    print(f"\nOverall Average CodeBLEU across all datasets per metric: {overall_avg}")


    for metric, value in overall_avg.items():
        logging.info(f"Overall {metric}: {value:.3f}")
        print(f"Overall {metric}: {value:.3f}")

    return all_dataset_scores, overall_avg

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate glue code generation with LLM")
    parser.add_argument("--n_shots", type=int, default=3, help="Number of few-shot examples")
    parser.add_argument("--num_steps", type=int, default=None, help="Number of steps to evaluate per dataset")
    args = parser.parse_args()

    # ----------- Set logging ------------
    LOG_PATH = f"logs/glue_code_generate_{args.n_shots}shot.log"
    if not os.path.exists(os.path.dirname(LOG_PATH)):
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

    logging.basicConfig(
        filename=LOG_PATH,
        filemode="w",
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # ----------- Run main process ------------
    asyncio.run(evaluate_all_datasets(DATASET_DIR, n_shots=args.n_shots, num_steps=args.num_steps))




