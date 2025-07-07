import pandas as pd
import json
import pyterrier as pt
import logging
import shutil
import re
import asyncio
from pathlib import Path
import torch
from rerankers import Reranker
from tqdm.asyncio import tqdm_asyncio
import os

from numerical.helper.run_config import RunConfig
# TODO:
# - This can be done a lot better with caching of results, storing of the indeex. Also, investigate query expansion
# - Consider cleaning up the corpus and the claims before IR step with a stemmer and other nltk niceties
# - Test and Handle GPU models for reranking gracefully


# Read more on reranking here: https://medium.com/@aniketpatil8451/understanding-rerankers-what-they-are-and-why-they-matter-in-machine-learning-bec045a808bd
class RetrieveRerank:
    def __init__(self):
        self._evidence_corpus = None

    def _init_reranker(self):
        # prepare the reranker
        gpu_mode = RunConfig.reranking["gpu"]
        logging.info(f"GPU Mode '{gpu_mode}'")
        if gpu_mode:
            self.model_name = RunConfig.reranking["model_name_gpu"]
            model_type = RunConfig.reranking["model_type_gpu"]
            if model_type == "rankllm":
                api_key = os.environ.get("OPENAI_API_KEY")
                batch_size = None  # rankllm only supports batch size of 1
                device = None
            else:
                api_key = None
                batch_size = RunConfig.reranking["batch_size"]
                device = torch.device(
                    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
                )

            self.ranker = Reranker(
                model_name=self.model_name,
                model_type=model_type,
                batch_size=batch_size,
                # device=device,
                # api_key=api_key,
            )
            logging.info(
                f"Initialized Reranker Model '{self.model_name}' with type rerank-type '{model_type}'."
                + f" Using device '{device}' and batch_size '{batch_size}'"
            )

        else:
            model_name = RunConfig.reranking["model_name_cpu"]
            self.ranker = Reranker(model_name)
            logging.info(f"Initialized Reranker Model '{model_name}' on device 'cpu'.")

    def _init_tokenizer(self):
        if not pt.java.started():
            pt.java.init()
        # We need this step to clean up some particularities with how python terrier handles special characters later on
        self.pt_tokenizer = pt.java.autoclass("org.terrier.indexing.tokenisation.Tokeniser").getTokeniser()

    def init(self, test: bool = False):
        # prepare corpus
        self._prepare_evidence_corpus(test)

        self._init_reranker()

        self._init_tokenizer()

        # prepare retriever
        self.retriever = self._prepare_retriever()

        self._init_reranker()

    def retrieve_stage(self, dataset: dict):
        logging.info("Retrieving query dataset from corpus index")

        # prepare data
        claims = self._prepare_claims(dataset).drop("query_0", axis=1)

        # run retriever
        retrieved_results = self._run_retriever(self.retriever, claims)
        return retrieved_results

    def rerank_stage(self, retrieved_results):
        logging.info("Reranking retrievied results")
        retrieved_results = retrieved_results.copy(deep=False)
        output = self._run_reranker(retrieved_results)
        return output

    # Wrapper function for running async function in a sync environment
    def _run_reranker(self, retrieved_results):
        return asyncio.run(self._rerank_stage_async(retrieved_results))

    async def _rerank_query_async(self, ranker, query, docnos, corpus):
        """Asynchronously reranks documents for a given query."""
        docs = corpus[corpus["docno"].isin(docnos)]["text"].tolist()
        doc_ids = corpus[corpus["docno"].isin(docnos)]["docno"].tolist()

        rerank_result = await ranker.rank_async(query=query, docs=docs, doc_ids=doc_ids)

        return [i.doc_id for i in rerank_result], [i.score for i in rerank_result]

    async def _rerank_stage_async(self, retrieved_results):
        """Runs reranking for multiple queries asynchronously."""
        tasks = [
            self._rerank_query_async(self.ranker, row["query"], row["docno"], self._evidence_corpus)
            for _, row in retrieved_results.iterrows()
        ]

        results = await tqdm_asyncio.gather(*tasks)

        # Unpacking results into separate lists
        retrieved_results.loc[:, "reranked_docno"] = [res[0] for res in results]
        retrieved_results.loc[:, "reranked_scores"] = [res[1] for res in results]

        return retrieved_results

    def _run_retriever(self, retriever, claims):
        """Run and post-process query retrieval results and store them all in one final dataframe"""
        retrieval_results = retriever.transform(claims).sort_values(by=["qid", "rank"])

        # Format results
        aggregated_results = (
            retrieval_results.groupby(["qid", "query", "claim_id", "type"])  # group by qid and query
            .agg({"docno": list})  # collect docnos as a list, docnos link to the corpus
            .reset_index()
        )

        return aggregated_results

    def _prepare_retriever(self):
        logging.debug(">>> RUNNING _prepare_retriever()")
        index_dir = Path(RunConfig.data["dir"]) / Path(RunConfig.data["retriever_index_path"])
        print(index_dir)
        if RunConfig.reranking["clean_index"]:
            logging.info(f"Deleting retriever index at '{index_dir}'")
            if index_dir.exists():
                shutil.rmtree(index_dir)

        logging.info(f"Indexing the corpus at '{index_dir}'")
        indexer = pt.IterDictIndexer(str(index_dir))
        index_ref = indexer.index(self.evidence_corpus.to_dict("records"))

        # Initialize the retriever according to the chosen word model
        wmodel = RunConfig.reranking["wmodel"]
        num_results = RunConfig.reranking["k"]
        logging.info(f"Initializing Retriever '{wmodel}' with num_results '{num_results}' given the corpus index")
        retriever = pt.terrier.Retriever(index_ref, wmodel=wmodel, num_results=num_results)  # there are other options here
        return retriever

    def _prepare_claims(self, claims_data: dict):
        list_claims = []
        for claim_idx, claim in enumerate(claims_data):
            # original claim
            list_claims.append((f"{claim['claim_id']}_orig", "original_claim", claim["original_claim"], claim["claim_id"]))

            # each question
            q_num = 0
            for key, value in claim.items():
                if key.startswith("question"):
                    qid_str = f"{claim['claim_id']}_q{q_num}"
                    list_claims.append((qid_str, key, value, claim["claim_id"]))
                    q_num += 1

        """need this for our data to make it work with pyterrier due to some weird characters"""
        output = pd.DataFrame(list_claims, columns=["qid", "type", "query", "claim_id"])
        output = pt.apply.query(lambda r: self._strip_markup(r.query))(output)
        return output

    def _strip_markup(self, text):
        """need this for our data to make it work with pyterrier due to some weird characters"""

        text = re.sub(r"[\\/]+", " ", text)
        return " ".join(self.pt_tokenizer.getTokens(text))

    def _prepare_evidence_corpus(self, test: bool = False):
        # needed for pyterrier
        if not test:
            evidence_corpus_path = Path(RunConfig.data["dir"]) / Path(RunConfig.data["evidence_corpus"])
        else:
            evidence_corpus_path = Path(RunConfig.data["dir"]) / Path(RunConfig.data["test_corpus"])
        logging.info(f"Reading evidence corpus at '{evidence_corpus_path}'")
        with open(evidence_corpus_path) as f:
            evidence_corpus = json.load(f)
        evidence_corpus = [{"docno": str(idx), "text": text} for idx, text in evidence_corpus.items()]
        self._evidence_corpus = pd.DataFrame(evidence_corpus)

    @property
    def evidence_corpus(self):
        return self._evidence_corpus
