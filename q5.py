"""
Q5 - Liar's Market probability calculator (WQC F26 packet, p.17).

Active cards = 3 face-up (given) + 2 face-down (unknown, drawn from the remaining 51 cards
of a 54-card deck: 52 + 2 jokers). Jokers are wild and always try to make the claimed hand.

Every probability is "the claimed hand exists among the 5 active cards" (what a "Bluff"
call checks), e.g. pair("6") with 6H 6S 10C face up = 1, whatever else is there.
Exception - high card: high_card("J") = P(the highest card present is a J), i.e. a J
(or a joker acting as one) and no natural Q, K or A among the active cards.

Usage:  python q5.py 2HJD8C        (T or 10 = ten, A = ace, RR = joker)

As a library:
    lm = LiarsMarket("2HJD8C")
    lm.pair("J"), lm.high_card("K"), lm.full_house("J", 8), lm.straight_flush("9", "H") ...
"""
import re
import sys
from fractions import Fraction
from itertools import combinations

SUITS = "SHDC"
RANK_CHARS = {"A": 14, "K": 13, "Q": 12, "J": 11, "T": 10}
RANK_NAMES = {14: "A", 13: "K", 12: "Q", 11: "J", 10: "10"}
RANKS = range(2, 15)          # 2..14, ace = 14
JOKER = "RR"


class Card:
    """rank: 2..14 (ace = 14), suit: one of SHDC. Joker: rank = -1, suit = 'R'."""

    def __init__(self, suit, rank):
        self.suit = suit
        self.rank = rank

    @property
    def is_joker(self):
        return self.rank == -1

    @classmethod
    def convert(cls, card_str):
        card_str = card_str.upper()
        if card_str == JOKER:
            return cls("R", -1)
        rank_s, suit = card_str[:-1], card_str[-1]
        if suit not in SUITS:
            raise ValueError(f"bad suit in {card_str!r}")
        return cls(suit, to_rank(rank_s))

    def key(self):
        return (self.rank, self.suit)

    def __repr__(self):
        return JOKER if self.is_joker else f"{rank_name(self.rank)}{self.suit}"


def rank_name(r):
    return RANK_NAMES.get(r, str(r))


def to_rank(r):
    """'J' / 'j' / 11 / '10' / 'T' -> 11 / 11 / 11 / 10 / 10. Ace = 14 (1 also accepted)."""
    if isinstance(r, int):
        rank = 14 if r == 1 else r
    else:
        s = str(r).upper()
        rank = RANK_CHARS.get(s, int(s) if s.isdigit() else -1)
        rank = 14 if rank == 1 else rank
    if rank not in RANKS:
        raise ValueError(f"bad rank {r!r}")
    return rank


def parse_cards(s):
    """Split e.g. '2HJD8C', '10HRRAS' or '2H JD 8C' into Card objects."""
    compact = s.upper().replace(" ", "")
    tokens = re.findall(r"RR|10[SHDC]|[2-9TJQKA][SHDC]", compact)
    if "".join(tokens) != compact:
        raise ValueError(f"could not parse cards from {s!r}")
    return [Card.convert(t) for t in tokens]


def full_deck():
    return [Card(s, r) for r in RANKS for s in SUITS] + [Card("R", -1), Card("R", -1)]


def remaining_deck(known):
    """Deck minus the known cards (each joker removes one joker)."""
    deck = full_deck()
    for c in known:
        for i, d in enumerate(deck):
            if d.key() == c.key():
                del deck[i]
                break
        else:
            raise ValueError(f"card {c} appears too many times")
    return deck


# ---------------- claims, listed from the best hand to the worst ----------------
# A claim is (class name, spec tuple). Its predicate says whether the active cards
# (rank counts c, set of (rank, suit) s, joker count j) can make it, jokers being wild.

CLASS_ORDER = ["Six-of-a-Kind", "Five-of-a-Kind", "Straight Flush", "Four-of-a-Kind",
               "Full House", "Three-of-a-Kind", "Two Pair", "Straight", "Pair", "High Card"]
HIGH_TO_LOW = list(range(14, 1, -1))


def straight_window(top):
    """Ranks in a straight with the given top card; the ace (14) also plays low in 5-high."""
    return [14 if r == 1 else r for r in range(top - 4, top + 1)]


def _kind(n, r):
    return lambda c, s, j: c[r] + j >= n


def _straight(w):
    return lambda c, s, j: sum(c[r] == 0 for r in w) <= j


def _straight_flush(w, su):
    return lambda c, s, j: sum((r, su) not in s for r in w) <= j


def _full_house(a, b):
    return lambda c, s, j: max(0, 3 - c[a]) + max(0, 2 - c[b]) <= j


def _two_pair(a, b):
    return lambda c, s, j: max(0, 2 - c[a]) + max(0, 2 - c[b]) <= j


def _high_card(r):
    # the highest card is r: an r (or a joker playing r) and no natural card above r
    return lambda c, s, j: c[r] + j >= 1 and all(c[x] == 0 for x in range(r + 1, 15))


def build_claims():
    claims = []
    for n, name in [(6, "Six-of-a-Kind"), (5, "Five-of-a-Kind")]:
        claims += [((name, (r,)), _kind(n, r)) for r in HIGH_TO_LOW]
    claims += [(("Straight Flush", (top, su)), _straight_flush(straight_window(top), su))
               for top in range(14, 4, -1) for su in SUITS]
    claims += [(("Four-of-a-Kind", (r,)), _kind(4, r)) for r in HIGH_TO_LOW]
    claims += [(("Full House", (a, b)), _full_house(a, b))
               for a in HIGH_TO_LOW for b in HIGH_TO_LOW if a != b]
    claims += [(("Three-of-a-Kind", (r,)), _kind(3, r)) for r in HIGH_TO_LOW]
    claims += [(("Two Pair", (a, b)), _two_pair(a, b))
               for a in HIGH_TO_LOW for b in HIGH_TO_LOW if a > b]
    claims += [(("Straight", (top,)), _straight(straight_window(top))) for top in range(14, 4, -1)]
    claims += [(("Pair", (r,)), _kind(2, r)) for r in HIGH_TO_LOW]
    claims += [(("High Card", (r,)), _high_card(r)) for r in HIGH_TO_LOW]
    return claims


CLAIMS = build_claims()


def existing_claims(active):
    """Every claim that exists among the active cards (jokers wild)."""
    c = {r: 0 for r in RANKS}
    s, j = set(), 0
    for card in active:
        if card.is_joker:
            j += 1
        else:
            c[card.rank] += 1
            s.add((card.rank, card.suit))
    return [claim for claim, pred in CLAIMS if pred(c, s, j)]


def describe(name, spec):
    if name == "Straight Flush":
        return f"{name}, {rank_name(spec[0])}-high {spec[1]}"
    if name == "Straight":
        return f"{name}, {rank_name(spec[0])}-high"
    if name == "Full House":
        return f"{name}, {rank_name(spec[0])}s over {rank_name(spec[1])}s"
    if name == "Two Pair":
        return f"{name}, {rank_name(spec[0])}s and {rank_name(spec[1])}s"
    if name == "High Card":
        return f"{name}, {rank_name(spec[0])}"
    return f"{name}, {rank_name(spec[0])}s"


class LiarsMarket:
    """Exact probabilities over every possible pair of face-down cards."""

    def __init__(self, face_up, n_hidden=2):
        self.face_up = parse_cards(face_up) if isinstance(face_up, str) else list(face_up)
        if len(self.face_up) != 3:
            raise ValueError(f"need exactly 3 face-up cards, got {self.face_up}")
        deck = remaining_deck(self.face_up)
        self.counts, self.class_counts, self.total = {}, {}, 0
        for hidden in combinations(deck, n_hidden):
            claims = existing_claims(self.face_up + list(hidden))
            for claim in claims:
                self.counts[claim] = self.counts.get(claim, 0) + 1
            for name in {n for n, _ in claims}:
                self.class_counts[name] = self.class_counts.get(name, 0) + 1
            self.total += 1

    def prob(self, name, *spec):
        return Fraction(self.counts.get((name, spec), 0), self.total)

    # ---- one method per hand class; ranks may be 'J', 11, 'T', '10', 'A', 1, ... ----
    def six_of_a_kind(self, rank):
        return self.prob("Six-of-a-Kind", to_rank(rank))

    def five_of_a_kind(self, rank):
        return self.prob("Five-of-a-Kind", to_rank(rank))

    def straight_flush(self, high_rank, suit):
        return self.prob("Straight Flush", to_rank(high_rank), suit.upper())

    def four_of_a_kind(self, rank):
        return self.prob("Four-of-a-Kind", to_rank(rank))

    def full_house(self, trips_rank, pair_rank):
        return self.prob("Full House", to_rank(trips_rank), to_rank(pair_rank))

    def three_of_a_kind(self, rank):
        return self.prob("Three-of-a-Kind", to_rank(rank))

    def two_pair(self, high_rank, low_rank):
        a, b = sorted((to_rank(high_rank), to_rank(low_rank)), reverse=True)
        return self.prob("Two Pair", a, b)

    def straight(self, high_rank):
        return self.prob("Straight", to_rank(high_rank))

    def pair(self, rank):
        return self.prob("Pair", to_rank(rank))

    def high_card(self, rank):
        return self.prob("High Card", to_rank(rank))

    # ---- summaries ----
    def class_prob(self, name):
        """P(at least one hand of this class exists)."""
        return Fraction(self.class_counts.get(name, 0), self.total)

    def best_in_class(self, name):
        """(probability, [specs]) of the most likely claim in a class (ties listed best-first)."""
        specs = [spec for (n, spec), _ in CLAIMS if n == name]
        probs = [self.prob(name, *spec) for spec in specs]
        best = max(probs)
        return best, [sp for sp, p in zip(specs, probs) if p == best]


def main():
    s = sys.argv[1] if len(sys.argv) > 1 else input("3 face-up cards (e.g. 2HJD8C): ").strip()
    lm = LiarsMarket(s)
    print(f"Face-up: {lm.face_up}   (2 face-down cards from the remaining 51)")
    print("P = probability that the claim exists among the 5 active cards (jokers wild)\n")
    print(f"{'#':>2}  {'Hand':<16} {'P(any)':>9} {'P(best claim)':>14}  Most likely claim")
    for i, name in enumerate(CLASS_ORDER, 1):
        p_best, specs = lm.best_in_class(name)
        p_class = float(lm.class_prob(name))
        if p_best == 0:
            print(f"{i:>2}. {name:<16} {p_class:>9.6f} {0:>14.6f}  impossible")
            continue
        extra = f"  (+{len(specs) - 1} tied)" if len(specs) > 1 else ""
        print(f"{i:>2}. {name:<16} {p_class:>9.6f} {float(p_best):>14.6f}  "
              f"{describe(name, specs[0])}{extra}")


if __name__ == "__main__":
    main()
