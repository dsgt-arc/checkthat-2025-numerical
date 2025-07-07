import polars as pl
import logging


def transform_to_factcheck(
    data: pl.DataFrame,
    top_n_evidences: int | None = None,
    top_n_evidences_refined: int | None = None,
    top_percentile_evidences: float | None = None,
    test: bool = False,
) -> list[str]:
    """Now we simply structure the training data, they call it 'features' here.
    Important: This is all put together as this is how you do it in NLP for claim verification!
    You put the CLAIM, QUESTIONS and EVIDENCE as one whole input set of words and classify that thing.
    Key word - context, due to attention.
    """
    if top_n_evidences:
        if top_percentile_evidences or top_n_evidences_refined:
            logging.error("Only one of top_n_evidences, top_n_evidences_refined or top_percentile_evidences can be provided.")
            raise ValueError("Only one of top_n_evidences, top_n_evidences_refined or top_percentile_evidences can be provided.")
        logging.info(f"Building factcheck format with {top_n_evidences} top_n_evidences.")
    elif top_percentile_evidences:
        if top_n_evidences or top_n_evidences_refined:
            logging.error("Only one of top_n_evidences, top_n_evidences_refined or top_percentile_evidences can be provided.")
            raise ValueError("Only one of top_n_evidences, top_n_evidences_refined or top_percentile_evidences can be provided.")
        logging.info(f"Building factcheck format with {top_percentile_evidences} top_percentile_evidences.")
    elif top_n_evidences_refined:
        if top_n_evidences or top_percentile_evidences:
            logging.error("Only one of top_n_evidences, top_n_evidences_refined or top_percentile_evidences can be provided.")
            raise ValueError("Only one of top_n_evidences, top_n_evidences_refined or top_percentile_evidences can be provided.")
        logging.info(f"Building factcheck format with {top_n_evidences_refined} top_n_evidences_refined.")
    else:
        logging.error("Either top_n_evidences, top_n_evidences_refined or top_percentile_evidences must be provided.")
        raise ValueError("Either top_n_evidences, top_n_evidences_refined or top_percentile_evidences must be provided.")

    # Filter the original claims
    claims = (
        data.filter(pl.col("type") == "original_claim")
        # maintain_order = maintains the order of col "claim_id". order in groups is maintained anyway
        .group_by("claim_id", maintain_order=True)
        .first()
    )
    if not test:
        claims = claims.select(pl.col("claim_id"), pl.col("query").alias("claim"), pl.col("label_encoded"))
    else:
        claims = claims.select(pl.col("claim_id"), pl.col("query").alias("claim"))

    if top_n_evidences:
        # Concat the questions and evidences
        questions_evidences = (
            # filter out the original claims
            data.filter(pl.col("type") != "original_claim")
            # add q_number for sorting
            .with_columns(pl.col("type").str.extract(r"(\d+)").alias("q_number"))
            # sort by greatest reranked_scores by top_n_evidences for each claim_id-type pair
            .sort(["claim_id", "q_number", "reranked_scores"], descending=[False, False, True])
            # remove duplicate evidence_text_reranker for the three different questions of the claim
            .unique(subset=["claim_id", "evidence_text_reranker"], keep="first", maintain_order=True)
            # get the top_n_evidences for each claim_id-question pair
            .group_by(["claim_id", "q_number"], maintain_order=True)
            .head(top_n_evidences)
            # merge the top evidences and questions together
            .group_by("claim_id", maintain_order=True)
            .all()
            .select(
                pl.col("claim_id"),
                pl.col("query").list.unique(maintain_order=True).list.join(separator="? ").alias("questions"),
                pl.col("evidence_text_reranker")
                .list.join(separator=" ")  # Evidences have already punctuation
                .alias("evidences"),
            )
        )

    elif top_n_evidences_refined:
        # Concat the questions and evidences
        questions_evidences = (
            # do not filter out original claims but filter out question2
            data.filter(pl.col("type") != "question2")
            # add q_number for sorting
            .with_columns(pl.col("type").str.extract(r"(\d+)").alias("q_number"))
            .with_columns(pl.col("q_number").fill_null(-1))
            # sort by greatest reranked_scores by top_n_evidences for each claim_id-type pair
            .sort(["claim_id", "q_number", "reranked_scores"], descending=[False, False, True])
            # get the top_n_evidences for each claim_id-question pair
            .group_by(["claim_id", "q_number"], maintain_order=True)
            .head(top_n_evidences_refined)
            # remove duplicate evidence_text_reranker in between the head
            .unique(subset=["claim_id", "evidence_text_reranker"], keep="first", maintain_order=True)
            # merge the top evidences and questions together
            .group_by("claim_id", maintain_order=True)
            .all()
            .select(
                pl.col("claim_id"),
                pl.col("query").list.unique(maintain_order=True).list.join(separator="? ").alias("questions"),
                pl.col("evidence_text_reranker")
                .list.join(separator=" ")  # Evidences have already punctuation
                .alias("evidences"),
            )
        )

    elif top_percentile_evidences:
        # Retrieve cutoff given percentile
        cutoff = (
            data
            # add q_number for sorting
            .with_columns(pl.col("type").str.extract(r"(\d+)").alias("q_number"))
            # sort by greatest reranked_scores by top_n_evidences for each claim_id-type pair
            .sort(["claim_id", "q_number", "reranked_scores"], descending=[False, False, True])
            # remove duplicate evidence_text_reranker for the three different questions of the claim
            .unique(subset=["claim_id", "evidence_text_reranker"], keep="first", maintain_order=True)
            # compute the cutoff per claim
            .group_by("claim_id", maintain_order=True)
            .agg(
                [
                    pl.col("reranked_scores").quantile(top_percentile_evidences, interpolation="nearest").alias("cutoff"),
                ]
            )
        )

        questions_evidences = (
            # filter out the original claims
            data
            # add q_number for sorting
            .with_columns(pl.col("type").str.extract(r"(\d+)").alias("q_number"))
            # sort by greatest reranked_scores by top_n_evidences for each claim_id-type pair
            .sort(["claim_id", "q_number", "reranked_scores"], descending=[False, False, True])
            # remove duplicate evidence_text_reranker for the three different questions of the claim
            # TODO: Move unique nach filter
            .unique(subset=["claim_id", "evidence_text_reranker"], keep="first", maintain_order=True)
            # filter out the evidences that are below the cutoff
            .join(cutoff, on=["claim_id"], how="left")
            .filter(pl.col("reranked_scores") >= pl.col("cutoff"))
            .group_by("claim_id", maintain_order=True)
            .all()
            .select(
                pl.col("claim_id"),
                pl.col("query").list.unique(maintain_order=True).list.join(separator="? ").alias("questions"),
                pl.col("evidence_text_reranker")
                .list.join(separator=" ")  # Evidences have already punctuation
                .alias("evidences"),
            )
        )

    # Join the claims and questions_evidences
    merged_data = claims.join(questions_evidences, on="claim_id")

    # Construct factcheck sequences
    factcheck_sequence = merged_data.with_columns(
        pl.Series(
            name="sequence",
            values="[Claim]: "
            + merged_data["claim"]
            + ". [Questions]: "
            + merged_data["questions"]
            + "? [Evidences]: "
            + merged_data["evidences"],
        ),
    )
    # Compute sequence length
    factcheck_sequence = factcheck_sequence.with_columns(
        pl.Series(name="sequence_length", values=factcheck_sequence["sequence"].str.len_chars())
    )

    sequence_stats = factcheck_sequence["sequence_length"].describe()
    logging.info(f"Sequence length statistics: {sequence_stats}")

    return factcheck_sequence


def transform_organizer_data_to_fc(data: pl.DataFrame) -> pl.DataFrame:
    """Transform the data from the organizer to factcheck format."""

    features = []
    for fact in data.iter_rows(named=True):
        claim = fact["claim"]
        evidences = []
        questions = []
        for question in fact["evidences"]:
            if len(question["top_k_doc"]) > 0:
                evidences.append(question["top_k_doc"][0])
            questions.append(question["questions"])
        questions = list(set(questions))
        evidences = list(set(evidences))
        feature = "[Claim]:" + claim + "[Questions]:" + " ".join(questions) + "[Evidences]:" + " ".join(evidences)
        features.append(feature)

    factcheck_sequence = data.with_columns(pl.Series(name="sequence", values=features))

    # Compute sequence length
    factcheck_sequence = factcheck_sequence.with_columns(
        pl.Series(name="sequence_length", values=factcheck_sequence["sequence"].str.len_chars())
    )

    sequence_stats = factcheck_sequence["sequence_length"].describe()
    logging.info(f"Sequence length statistics: {sequence_stats}")
    return factcheck_sequence
