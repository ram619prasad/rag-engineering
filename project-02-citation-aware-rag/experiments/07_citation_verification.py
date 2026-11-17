from project_02_citation_aware_rag.models import Claim
from project_02_citation_aware_rag.verification import verify_claim

evidence = (
    "We constructed our seed lexicon consisting of "
    "15 positive words and 15 negative words."
)

claims = [
    Claim(
        text="The seed lexicon contains 500 positive words.",
        citation_ids=["1909.00694:11:4"],
    ),
    Claim(
        text="The seed lexicon was created in 2015.",
        citation_ids=["1909.00694:11:4"],
    ),
]

# claim = Claim(
#     text="The seed lexicon contains 30 words.",
#     citation_ids=["1909.00694:11:4"],
# )
for claim in claims:
    result = verify_claim(claim, evidence)
    print(result.model_dump_json(indent=2))