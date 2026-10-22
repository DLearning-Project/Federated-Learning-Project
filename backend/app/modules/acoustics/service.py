"""Toy FedAvg aggregation.

Extension points (module owner): PORTIA dataset loading, cached spectrogram
generation, EfficientNet-B0 local training and measured evaluation results.
Real training must run in a separate worker (not inside an API request) and
must keep evaluation data separate from client training data.
"""

from .schemas import ClientWeight, FedAvgInput, FedAvgResponse, FedAvgResult

UNITS = {
    "total_samples": "samples",
    "weight": "fraction of total samples (sums to 1)",
    "global_vector": "same units as the client parameter vectors",
    "unweighted_mean": "same units as the client parameter vectors",
}

ASSUMPTIONS = [
    "Vectors are small toy parameters, not real EfficientNet-B0 weights.",
    "Each client's weight is n_k / Σ n: its share of all training samples.",
    "Global vector = Σ_k weight_k · vector_k (one FedAvg aggregation step).",
    "Clients with zero samples receive zero weight.",
    "No CNN is trained and no classification accuracy is predicted.",
]


def aggregate(inp: FedAvgInput) -> FedAvgResponse:
    total = sum(c.sample_count for c in inp.clients)
    dim = len(inp.clients[0].vector)
    weights = [c.sample_count / total for c in inp.clients]

    global_vector = [
        sum(w * c.vector[i] for w, c in zip(weights, inp.clients)) for i in range(dim)
    ]
    unweighted = [sum(c.vector[i] for c in inp.clients) / len(inp.clients) for i in range(dim)]

    results = FedAvgResult(
        total_samples=total,
        weights=[
            ClientWeight(name=c.name, sample_count=c.sample_count, weight=w)
            for c, w in zip(inp.clients, weights)
        ],
        global_vector=global_vector,
        unweighted_mean=unweighted,
    )
    return FedAvgResponse(inputs=inp, results=results, units=UNITS, assumptions=ASSUMPTIONS)
