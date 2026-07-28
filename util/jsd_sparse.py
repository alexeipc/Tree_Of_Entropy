import math
from vllm import SamplingParams


def topk_to_probs(topk_dict):
    """
    vLLM top-K logprobs -> normalized probability dict

    Input:
        {
            token_id: Logprob(...),
            ...
        }

    Output:
        {
            token_id: probability,
            ...
        }
    """
    probs = {}

    for token_id, lp in topk_dict.items():
        probs[token_id] = math.exp(lp.logprob)

    Z = sum(probs.values())

    if Z == 0:
        return {}

    return {k: v / Z for k, v in probs.items()}


def jsd_sparse(P, Q):
    """
    Jensen-Shannon divergence between sparse distributions.
    """

    vocab = set(P.keys()) | set(Q.keys())

    M = {}
    for tok in vocab:
        M[tok] = 0.5 * P.get(tok, 0.0) + 0.5 * Q.get(tok, 0.0)

    def kl(A, B):
        s = 0.0

        for tok, p in A.items():
            if p > 0:
                s += p * math.log(p / B[tok])

        return s

    return 0.5 * kl(P, M) + 0.5 * kl(Q, M)


def future_disagreement(completions, horizon=10):
    """
    completions:
        outputs[0].outputs from vLLM

    Returns:
        average pairwise JSD over the future horizon
    """

    n = len(completions)

    total = 0.0
    count = 0

    for i in range(n):
        for j in range(i + 1, n):

            pair_jsd = 0.0
            steps = 0

            T = min(
                horizon,
                len(completions[i].logprobs),
                len(completions[j].logprobs),
            )

            for t in range(T):
                P = topk_to_probs(completions[i].logprobs[t])
                Q = topk_to_probs(completions[j].logprobs[t])

                pair_jsd += jsd_sparse(P, Q)
                steps += 1

            if steps > 0:
                pair_jsd /= steps

                total += pair_jsd
                count += 1

    if count == 0:
        return 0.0

    return total / count