"""
Sparse High-Order 5-gram Language Model with (Modified) Kneser-Ney Smoothing
=============================================================================

Built entirely from scratch (no NLTK / KenLM / SRILM) for a search-engine
autocomplete / query-prediction system.

Design summary
--------------
1. SPARSE STORAGE
   Counts are stored only for n-grams that were actually observed, using
   nested dicts:  counts[order][context_tuple][last_word] = count
   This avoids the O(V^n) dense matrix that a vocabulary of millions of
   query terms would otherwise require.

2. RECURSIVE INTERPOLATED KNESER-NEY SMOOTHING
   - The *highest* order (5-gram) uses raw counts.
   - Every *lower* order (used only as a back-off distribution) uses
     "continuation counts" -- the number of DISTINCT single-word contexts
     that precede a given n-gram -- exactly as in Kneser-Ney theory,
     rather than raw counts. This is what lets the model generalize to
     n-grams it has never seen as the leading part of a longer sequence.
   - A single discount D per order is estimated from the data with the
     classic Good-Turing-style formula   D = n1 / (n1 + 2*n2)
     (n1 = count of n-grams seen exactly once, n2 = count of n-grams seen
     exactly twice), which is the standard way KenLM/SRILM pick D as well.
   - probability(word | context) is computed *recursively*:
         P(w|c_1..c_{k-1}) = discounted_term + lambda(context) * P(w|c_2..c_{k-1})
     bottoming out at a uniform/OOV distribution over the vocabulary.

3. OOV / DYNAMIC VOCABULARY HANDLING
   - Rare training words are collapsed into <UNK> up front, so the model
     always has *some* probability mass for tokens it has never seen.
   - `add_trending_terms()` lets brand-new product names, place names, or
     trending hashtags be *injected* into the vocabulary and given a small
     "seed" count without a full retrain -- important for a live search
     product where new entities appear every hour.
   - `update_from_queries()` performs incremental (online) count updates
     from a stream of freshly logged queries and lazily refreshes the
     Kneser-Ney discounts.

4. REAL-TIME AUTOCOMPLETE
   - `predict_next_word()` returns the top-k most probable next tokens
     for a typed prefix using the recursive back-off probability.
   - `autocomplete()` extends this into full multi-word suggestions
     (e.g. "in India", "near me") via greedy/beam expansion.

This is an educational/production-pattern reference implementation -- it
trades a few of the finer details of "modified" KN (which uses 3 separate
discounts D1, D2, D3+ per order) for a single per-order discount, which is
the classic (non-modified) interpolated Kneser-Ney formulation. The code
is structured so that swapping in 3-way discounting later is a small,
localized change (see `_estimate_discount`).
"""

import math
import re
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Optional

UNK = "<UNK>"
BOS = "<s>"
EOS = "</s>"
MAX_ORDER = 5


def tokenize(text: str) -> List[str]:
    """Very small, dependency-free tokenizer: lowercase + strip punctuation."""
    text = text.lower().strip()
    tokens = re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text)
    return tokens


class SparseKNLanguageModel:
    def __init__(self, max_order: int = MAX_ORDER, unk_threshold: int = 1):
        self.max_order = max_order
        self.unk_threshold = unk_threshold

        # counts[k][context_tuple][word] = raw count of the k-gram
        # context_tuple has length k-1 ; k ranges 1..max_order
        self.counts: Dict[int, Dict[Tuple[str, ...], Counter]] = {
            k: defaultdict(Counter) for k in range(1, max_order + 1)
        }

        # continuation predecessor sets, used to derive continuation counts.
        # cont_preds[k][ (t1..tk) ] = set of distinct tokens t0 that occurred
        # immediately before the k-gram (t1..tk) somewhere in the corpus.
        # This is exactly what Kneser-Ney needs for the *lower-order*
        # back-off distributions.
        self.cont_preds: Dict[int, Dict[Tuple[str, ...], set]] = {
            k: defaultdict(set) for k in range(1, max_order + 1)
        }

        self.vocab: Counter = Counter()
        self.discounts: Dict[int, float] = {}
        self._trained = False

    # ------------------------------------------------------------------ #
    # Training / incremental updates
    # ------------------------------------------------------------------ #
    def train(self, queries: List[str]) -> None:
        raw_vocab = Counter()
        for q in queries:
            raw_vocab.update(tokenize(q))

        # words seen <= unk_threshold times are folded into <UNK> so the
        # model has genuine probability mass for out-of-vocabulary tokens.
        self.vocab = Counter(
            {w: c for w, c in raw_vocab.items() if c > self.unk_threshold}
        )
        self.vocab[UNK] += sum(c for w, c in raw_vocab.items() if c <= self.unk_threshold) or 1

        self._ingest(queries)
        self._recompute_discounts()
        self._trained = True

    def update_from_queries(self, new_queries: List[str]) -> None:
        """Incremental / online update from freshly logged search queries,
        without discarding previously learned statistics."""
        for q in new_queries:
            for tok in tokenize(q):
                if tok not in self.vocab and self.vocab[UNK] == 0:
                    self.vocab[UNK] = 1
        self._ingest(new_queries)
        self._recompute_discounts()

    def add_trending_terms(self, terms: List[str], seed_count: int = 3) -> None:
        """Inject brand-new trending words (new product names, events,
        locations) into the live vocabulary with a small seed count so
        they can immediately participate in back-off scoring, without a
        full retrain."""
        for t in terms:
            t = t.lower().strip()
            if t:
                self.vocab[t] += seed_count

    def _normalize(self, token: str) -> str:
        return token if token in self.vocab else UNK

    def _ingest(self, queries: List[str]) -> None:
        for q in queries:
            tokens = tokenize(q)
            if not tokens:
                continue
            tokens = [self._normalize(t) for t in tokens]
            # pad with BOS/EOS so short queries still contribute low-order stats
            padded = [BOS] * (self.max_order - 1) + tokens + [EOS]

            for k in range(1, self.max_order + 1):
                for i in range(len(padded) - k + 1):
                    gram = tuple(padded[i:i + k])
                    context, word = gram[:-1], gram[-1]
                    self.counts[k][context][word] += 1
                    if i > 0:
                        predecessor = padded[i - 1]
                        self.cont_preds[k][gram].add(predecessor)

    # ------------------------------------------------------------------ #
    # Kneser-Ney discount estimation
    # ------------------------------------------------------------------ #
    def _estimate_discount(self, k: int) -> float:
        """D = n1 / (n1 + 2*n2) using counts of this order (the standard
        single-discount interpolated-KN estimator)."""
        n1 = n2 = 0
        for ctx_counter in self.counts[k].values():
            for c in ctx_counter.values():
                if c == 1:
                    n1 += 1
                elif c == 2:
                    n2 += 1
        if n1 == 0 or n2 == 0:
            return 0.75  # safe, widely-used default fallback
        d = n1 / (n1 + 2 * n2)
        return min(max(d, 0.1), 0.9)  # keep it in a sane range

    def _recompute_discounts(self) -> None:
        for k in range(1, self.max_order + 1):
            self.discounts[k] = self._estimate_discount(k)

    # ------------------------------------------------------------------ #
    # Continuation-count helpers (lower-order KN distribution)
    # ------------------------------------------------------------------ #
    def _continuation_count(self, k: int, gram: Tuple[str, ...]) -> int:
        """Number of distinct single-word contexts preceding this k-gram."""
        return len(self.cont_preds[k].get(gram, ()))

    def _continuation_context_total(self, k: int, context: Tuple[str, ...]) -> int:
        """Sum, over all words w observed with this context at order k, of
        their continuation counts. This is the denominator for the
        continuation ("lower-order") KN distribution."""
        total = 0
        for w in self.counts[k].get(context, {}):
            total += self._continuation_count(k, context + (w,))
        return total

    # ------------------------------------------------------------------ #
    # Recursive Kneser-Ney probability
    # ------------------------------------------------------------------ #
    def probability(self, word: str, context: Tuple[str, ...]) -> float:
        """P(word | context), recursively backing off from order
        len(context)+1 down to the unigram / uniform-OOV base case."""
        word = self._normalize(word)
        context = tuple(self._normalize(c) for c in context)
        order = len(context) + 1
        order = min(order, self.max_order)
        context = context[-(order - 1):] if order > 1 else ()
        return self._kn_prob(word, context, order)

    def _kn_prob(self, word: str, context: Tuple[str, ...], order: int) -> float:
        if order == 0:
            # base case: uniform distribution over vocabulary (handles
            # genuinely novel OOV combinations gracefully)
            V = max(len(self.vocab), 1)
            return 1.0 / V

        D = self.discounts.get(order, 0.75)

        if order == self.max_order:
            # top order: use RAW counts (standard interpolated KN)
            ctx_counter = self.counts[order].get(context, {})
            context_total = sum(ctx_counter.values())
            if context_total == 0:
                # never saw this context at all -> fall back immediately
                return self._kn_prob(word, context[1:], order - 1)
            count_w = ctx_counter.get(word, 0)
            discounted = max(count_w - D, 0) / context_total
            n_distinct = len(ctx_counter)
            lam = (D * n_distinct) / context_total
            lower = self._kn_prob(word, context[1:], order - 1)
            return discounted + lam * lower
        else:
            # lower order used purely as a back-off distribution:
            # use CONTINUATION counts, the heart of Kneser-Ney.
            ctx_counter = self.counts[order].get(context, {})
            cont_total = self._continuation_context_total(order, context)
            if cont_total == 0:
                return self._kn_prob(word, context[1:], order - 1)
            cw = self._continuation_count(order, context + (word,))
            discounted = max(cw - D, 0) / cont_total
            n_distinct = len(ctx_counter)
            lam = (D * n_distinct) / cont_total if cont_total > 0 else 0.0
            lower = self._kn_prob(word, context[1:], order - 1)
            return discounted + lam * lower

    # ------------------------------------------------------------------ #
    # Autocomplete-facing API
    # ------------------------------------------------------------------ #
    def predict_next_word(
        self, prefix_text: str, top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Return the top-k most likely next tokens for a typed prefix."""
        tokens = [self._normalize(t) for t in tokenize(prefix_text)]
        context = tuple((([BOS] * (self.max_order - 1)) + tokens)[-(self.max_order - 1):])

        # Candidate words: prefer words actually observed following ANY
        # suffix of the context (keeps candidate set small & relevant,
        # which matters at web scale); fall back to whole vocabulary only
        # if nothing was observed.
        candidates = set()
        for k in range(self.max_order, 0, -1):
            sub_ctx = context[-(k - 1):] if k > 1 else ()
            observed = self.counts[k].get(sub_ctx)
            if observed:
                candidates.update(observed.keys())
        if not candidates:
            candidates = set(self.vocab.keys())
        candidates.discard(BOS)

        scored = [(w, self.probability(w, context)) for w in candidates if w != UNK]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def autocomplete(
        self, prefix_text: str, num_words: int = 3, top_k: int = 5
    ) -> List[str]:
        """Generate up to top_k full multi-word suggestions extending the
        typed prefix (e.g. 'best places to visit' -> 'in india', 'near me').
        Uses greedy beam expansion driven by the recursive KN probabilities.
        """
        beams: List[Tuple[List[str], float]] = [([], 0.0)]
        for _ in range(num_words):
            new_beams = []
            for words_so_far, log_p in beams:
                extended_prefix = (prefix_text + " " + " ".join(words_so_far)).strip()
                next_words = self.predict_next_word(extended_prefix, top_k=top_k)
                for w, p in next_words:
                    if w == EOS:
                        new_beams.append((words_so_far, log_p))
                        continue
                    new_beams.append((words_so_far + [w], log_p + math.log(p + 1e-12)))
            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[:top_k]

        suggestions, seen = [], set()
        for words, _ in beams:
            phrase = " ".join(words).strip()
            if phrase and phrase not in seen:
                seen.add(phrase)
                suggestions.append(phrase)
        return suggestions[:top_k]


# ------------------------------------------------------------------------- #
# Demo / smoke test
# ------------------------------------------------------------------------- #
if __name__ == "__main__":
    corpus = [
        "best places to visit in India",
        "best places to visit near me",
        "best places to visit in Chennai",
        "best places to visit during summer",
        "best places to visit in India during winter",
        "best places to eat in Chennai",
        "best places to eat near me",
        "best restaurants to visit in Chennai",
        "top places to visit in India",
        "top places to visit in Tamil Nadu",
        "cheap flights to Chennai",
        "cheap flights to Mumbai",
        "weather in Chennai today",
        "weather in Mumbai today",
        "best places to visit in Kerala during monsoon",
        "best places to visit with family in India",
        "best places to visit alone in India",
    ] * 5  # repeat to give the smoothing model something to chew on

    model = SparseKNLanguageModel(max_order=5, unk_threshold=1)
    model.train(corpus)

    print("=== Next-word predictions for 'best places to visit' ===")
    for w, p in model.predict_next_word("best places to visit", top_k=5):
        print(f"  {w:12s}  P={p:.4f}")

    print("\n=== Full autocomplete suggestions ===")
    for s in model.autocomplete("best places to visit", num_words=2, top_k=5):
        print(f"  best places to visit {s}")

    print("\n=== Handling a brand-new trending term (OOV) ===")
    model.add_trending_terms(["cyberpunk2099", "vizag"])
    model.update_from_queries(["best places to visit in vizag", "best places to visit in vizag during summer"])
    for w, p in model.predict_next_word("best places to visit in", top_k=5):
        print(f"  {w:12s}  P={p:.4f}")

    print("\nEstimated per-order KN discounts:", model.discounts)
