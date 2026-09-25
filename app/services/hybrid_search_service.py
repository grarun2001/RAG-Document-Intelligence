from app.services.retrieval_service import (
    retrieve_chunks
)

from app.services.keyword_search_service import (
    keyword_search
)


# --------------------------------------------------
# Hybrid Search Configuration
# --------------------------------------------------

SEMANTIC_TOP_K = 10
KEYWORD_TOP_K = 10
HYBRID_TOP_K = 8

RRF_K = 60


# --------------------------------------------------
# Reciprocal Rank Fusion
# --------------------------------------------------

def reciprocal_rank_fusion(
    semantic_results: list[dict],
    keyword_results: list[dict],
    top_k: int = HYBRID_TOP_K
) -> list[dict]:
    """
    Combine semantic and keyword search results
    using Reciprocal Rank Fusion (RRF).

    Each chunk is uniquely identified using:
        (document_id, chunk_id)
    """

    combined = {}

    # ----------------------------------------------
    # 1. Semantic Results
    # ----------------------------------------------

    for rank, result in enumerate(
        semantic_results,
        start=1
    ):

        key = (
            result["document_id"],
            result["chunk_id"]
        )

        if key not in combined:

            combined[key] = {
                **result,

                "semantic_rank": rank,

                "keyword_rank": None,

                "semantic_score": result.get(
                    "score",
                    0.0
                ),

                "keyword_score": 0.0,

                "rrf_score": 0.0
            }

        combined[key]["rrf_score"] += (
            1 / (RRF_K + rank)
        )

    # ----------------------------------------------
    # 2. Keyword Results
    # ----------------------------------------------

    for rank, result in enumerate(
        keyword_results,
        start=1
    ):

        key = (
            result["document_id"],
            result["chunk_id"]
        )

        if key not in combined:

            combined[key] = {
                **result,

                "semantic_rank": None,

                "keyword_rank": rank,

                "semantic_score": 0.0,

                "keyword_score": result.get(
                    "keyword_score",
                    0.0
                ),

                "rrf_score": 0.0
            }

        else:

            combined[key]["keyword_rank"] = rank

            combined[key]["keyword_score"] = (
                result.get(
                    "keyword_score",
                    0.0
                )
            )

        combined[key]["rrf_score"] += (
            1 / (RRF_K + rank)
        )

    # ----------------------------------------------
    # 3. Sort by RRF Score
    # ----------------------------------------------

    results = list(
        combined.values()
    )

    results.sort(
        key=lambda x: x["rrf_score"],
        reverse=True
    )

    # ----------------------------------------------
    # 4. Return Top K
    # ----------------------------------------------

    return results[:top_k]


# --------------------------------------------------
# Main Hybrid Search
# --------------------------------------------------

def hybrid_search(
    question: str,
    document_id: str | None = None,
    top_k: int = HYBRID_TOP_K
) -> list[dict]:
    """
    Perform:

        Semantic Search
              +
        Keyword Search
              ↓
        Reciprocal Rank Fusion
              ↓
        Final Hybrid Results
    """

    # ----------------------------------------------
    # Validate Question
    # ----------------------------------------------

    if not question.strip():

        raise ValueError(
            "Question cannot be empty."
        )

    # ----------------------------------------------
    # 1. Semantic Search
    # ----------------------------------------------

    semantic_results = retrieve_chunks(
        question=question,
        document_id=document_id,
        top_k=SEMANTIC_TOP_K
    )

    # ----------------------------------------------
    # 2. Keyword Search
    # ----------------------------------------------

    keyword_results = keyword_search(
        question=question,
        document_id=document_id,
        top_k=KEYWORD_TOP_K
    )

    # ----------------------------------------------
    # 3. Hybrid Fusion
    # ----------------------------------------------

    hybrid_results = reciprocal_rank_fusion(
        semantic_results=semantic_results,
        keyword_results=keyword_results,
        top_k=top_k
    )

    return hybrid_results


# --------------------------------------------------
# Test
# --------------------------------------------------

if __name__ == "__main__":

    question = (
        "What types of leave are available "
        "to employees?"
    )

    print(
        "\nRunning hybrid search..."
    )

    try:

        results = hybrid_search(
            question=question,
            top_k=5
        )

        print(
            f"\nFinal results: {len(results)}"
        )

        for index, result in enumerate(
            results,
            start=1
        ):

            print(
                "\n" + "=" * 70
            )

            print(
                f"Rank          : {index}"
            )

            print(
                f"Document      : "
                f"{result.get('document_name')}"
            )

            print(
                f"Document ID   : "
                f"{result.get('document_id')}"
            )

            print(
                f"Page          : "
                f"{result.get('page_number')}"
            )

            print(
                f"Chunk ID       : "
                f"{result.get('chunk_id')}"
            )

            print(
                f"RRF Score     : "
                f"{result.get('rrf_score', 0.0):.6f}"
            )

            print(
                f"Semantic Rank : "
                f"{result.get('semantic_rank')}"
            )

            print(
                f"Keyword Rank  : "
                f"{result.get('keyword_rank')}"
            )

            print(
                f"Semantic Score: "
                f"{result.get('semantic_score', 0.0):.6f}"
            )

            print(
                f"Keyword Score : "
                f"{result.get('keyword_score', 0.0):.6f}"
            )

            print(
                "=" * 70
            )

            print(
                result.get(
                    "text",
                    ""
                )[:500]
            )

    except Exception as error:

        print(
            f"\nHybrid search failed: {error}"
        )