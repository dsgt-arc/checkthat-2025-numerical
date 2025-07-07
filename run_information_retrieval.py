from numerical.helper.run_config import RunConfig
from numerical.models.reranker import RetrieveRerank
from numerical.helper.logger import set_up_log
from pathlib import Path
import logging
import argparse
import polars as pl
import json


def init_args_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Information Retrieval Pipeline")

    parser.add_argument("--config_path", type=str, default="config/reranker/run_config_reranker_local.yml", help="Config name")
    parser.add_argument("--test-only", "-t", dest="test_only", action="store_true", default=False, help="Run only the test set")
    return parser.parse_args()


def consolidate_retrieved_results(
    reranking_results: pl.DataFrame, decomposed_claims: pl.DataFrame, evidence_corpus: pl.DataFrame
) -> pl.DataFrame:
    """Prepare retrieved data."""

    # Prepare reranking_results: mostly cleaning strings and casting to lists/arrays
    reranking_results = reranking_results.with_columns(
        [
            pl.col("reranked_docno").cast(pl.List(pl.Int64)),
            pl.col("docno").cast(pl.List(pl.Int64)),
        ]
    )

    # Merge reranked results with decomposed_claims to get the labels (and metadata)
    merged = reranking_results.join(decomposed_claims.select(["claim_id", "label"]), on="claim_id", how="left")

    # explode the retrieval (docno), reranking (reranked_docno, reranked_scores) columns
    merged = merged.explode(["docno", "reranked_docno", "reranked_scores"])

    # Rename evidence_corpus columns
    evidence_corpus = evidence_corpus.rename({"text": "evidence_text"})

    # Merge with evidence_corpus to get the text (for both retrieval and reranking -> compare incremental benefit of reranking)
    merged = merged.join(evidence_corpus, on="docno", how="left", validate="m:m").rename(
        {"evidence_text": "evidence_text_retriever"}
    )
    merged = merged.join(evidence_corpus, left_on="reranked_docno", right_on="docno", how="left", validate="m:m").rename(
        {"evidence_text": "evidence_text_reranker"}
    )

    return merged


def main():
    set_up_log()
    logging.info("Start Information Retrieval")
    try:
        args = init_args_parser()
        logging.info(f"Reading config '{args.config_path}'")
        rc = RunConfig()
        rc.load_config(args.config_path)

        if not args.test_only:
            logging.info(f"Starting Reranker with config: '{rc.reranking}'")
            information_retriever = RetrieveRerank()
            information_retriever.init(test=False)

            logging.info("Export prepared evidence corpus with retriever id's to match the retrieval data after")
            ir_corpus_path = Path(rc.data["dir"]) / Path(rc.data["ir_evidence_corpus"])
            ir_corpus_path.parent.mkdir(exist_ok=True)
            evidence_corpus = pl.DataFrame(information_retriever.evidence_corpus)
            evidence_corpus = evidence_corpus.with_columns(pl.col("docno").cast(pl.Int64))
            evidence_corpus.write_csv(ir_corpus_path)

            # retrieve and rerank

            # Validation set
            logging.info("Retrieval &  Reranking w/ validation dataset")
            dec_claims_val_path = Path(rc.data["dir"]) / Path(rc.data["claims_val"])
            with open(dec_claims_val_path) as f:
                dec_claims_val = json.load(f)

            retrieved_results_val = information_retriever.retrieve_stage(dataset=dec_claims_val)

            # Additional filtering out of questions
            if rc.reranking["remove_questions"]:
                retrieved_results_val = retrieved_results_val[
                    ~retrieved_results_val["type"].isin(rc.reranking["remove_questions"])
                ]

            reranked_results_val = pl.DataFrame(information_retriever.rerank_stage(retrieved_results_val))

            cons_results_val = consolidate_retrieved_results(
                reranked_results_val, pl.DataFrame(reranked_results_val), evidence_corpus
            )
            val_results_path = Path(rc.data["dir"]) / Path(rc.data["reranked_results_val"])
            logging.info(f"Saving retrieve-rerank result of validation dataset to '{val_results_path}'")
            cons_results_val.write_csv(val_results_path)

            # Train set
            logging.info("Retrieval &  Reranking w/ train dataset")
            dec_claims_train_path = Path(rc.data["dir"]) / Path(rc.data["claims_train"])
            with open(dec_claims_train_path) as f:
                dec_claims_train = json.load(f)
            retrieved_results_train = information_retriever.retrieve_stage(dataset=dec_claims_train)
            reranked_results_train = pl.DataFrame(information_retriever.rerank_stage(retrieved_results_train))

            cons_results_train = consolidate_retrieved_results(
                reranked_results_train, pl.DataFrame(dec_claims_train), evidence_corpus
            )
            train_results_path = Path(rc.data["dir"]) / Path(rc.data["reranked_results_train"])
            logging.info(f"Saving retrieve-rerank result of train dataset to '{train_results_path}'")
            cons_results_train.write_csv(train_results_path)

        # Test set

        logging.info(f"Starting Reranker with config: '{rc.reranking}'")
        information_retriever = RetrieveRerank()
        information_retriever.init(test=True)

        logging.info("Export prepared evidence corpus with retriever id's to match the retrieval data after")
        ir_corpus_path = Path(rc.data["dir"]) / Path(rc.data["ir_evidence_corpus_test"])
        ir_corpus_path.parent.mkdir(exist_ok=True)
        evidence_corpus = pl.DataFrame(information_retriever.evidence_corpus)
        evidence_corpus = evidence_corpus.with_columns(pl.col("docno").cast(pl.Int64))
        evidence_corpus.write_csv(ir_corpus_path)

        logging.info("Retrieval &  Reranking w/ test dataset")
        dec_claims_test_path = Path(rc.data["dir"]) / Path(rc.data["claims_test"])
        with open(dec_claims_test_path) as f:
            dec_claims_test = json.load(f)

        # rewrite the test claims to the same format as the train and val claims
        rewritten_claims = []
        for i, example in enumerate(dec_claims_test):
            rewritten_example = {}
            rewritten_example["claim_id"] = i
            for type, text in example.items():
                if type == "claim":
                    rewritten_example["original_claim"] = text.replace("\n", " ")
                    rewritten_example["label"] = None
                elif type == "questions":
                    for j, q in enumerate(text):
                        rewritten_example[f"question{j}"] = q["questions"].replace("\n", " ")

            rewritten_claims.append(rewritten_example)

        retrieved_results_test = information_retriever.retrieve_stage(dataset=rewritten_claims)

        # Additional filtering out of questions
        if rc.reranking["remove_questions"]:
            retrieved_results_test = retrieved_results_test[
                ~retrieved_results_test["type"].isin(rc.reranking["remove_questions"])
            ]

        reranked_results_test = pl.DataFrame(information_retriever.rerank_stage(retrieved_results_test))

        cons_results_test = consolidate_retrieved_results(reranked_results_test, pl.DataFrame(rewritten_claims), evidence_corpus)
        val_results_path = Path(rc.data["dir"]) / Path(rc.data["reranked_results_test"])
        logging.info(f"Saving retrieve-rerank result of validation dataset to '{val_results_path}'")
        cons_results_test.write_csv(val_results_path)

        logging.info("Finished Information Retrieval Pipeline.")
        return 0
    except Exception:
        logging.exception("Information Retrieval Pipeline failed", stack_info=True)
        return 1


main()
