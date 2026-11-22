"""
Q4 - Treasures & Trash hand improvement (WQC F26 packet, p.14-15).

Given a 9-card hand and the treasure suit, for each K find the top-10 sets of K new cards
(drawn from the remaining pool), each set listed once together with the K cards it is
best to remove for it.

Pool (360 cards, 10 decks, no jokers):
  treasure suit : 10 of every rank
  neutral suits : 7 of every rank, minus one 2 and one 3   (each of the two suits)
  trash suit    : 4 of every rank
"Remaining" = pool minus the current hand (other teams' hands are unknown).

Scoring per card: base (2-10 -> 10, J-K -> 30, A -> 50) x multiplier, multiplier starts at 1:
  treasure +1; trash -1 (or +1 if the whole hand is trash);
  rank appears exactly twice +0.5; three or more times +1.

Neutral suits score identically, so the search treats them as one type; a new card
shown as e.g. "7[S|C]" can be either neutral suit.

Usage:  python q4.py <9 cards> <treasure suit>    e.g.  python q4.py 2H3H5S7C9DJHQSKDAC H
"""
import sys
from itertools import combinations, combinations_with_replacement
import random 
import numpy as np

from q5 import SUITS, rank_name, parse_cards

RANKS = list(range(2, 15))
BASE = np.array([10 if r <= 10 else (50 if r == 14 else 30) for r in RANKS], dtype=np.float64)
TREASURE, NEUTRAL, TRASH = 0, 1, 2
CAT_NAME = {TREASURE: "treasure", NEUTRAL: "neutral", TRASH: "trash"}
NTYPES = 13 * 3                       # type index = cat * 13 + (rank - 2)
TOP_N = 10
SAME_COLOUR = {"H": "D", "D": "H", "S": "C", "C": "S"}


def suit_roles(treasure):
    trash = SAME_COLOUR[treasure]
    neutral = [s for s in SUITS if s not in (treasure, trash)]
    return trash, neutral


def pool_counts():
    c = np.zeros((3, 13), dtype=np.int64)
    c[TREASURE] = 10
    c[NEUTRAL] = 14
    c[NEUTRAL, 0] -= 2                 # one 2 removed from each neutral suit
    c[NEUTRAL, 1] -= 2                 # one 3 removed from each neutral suit
    c[TRASH] = 4
    return c.reshape(-1)


def score(hands):
    """hands: (N, 39) counts -> (N,) scores."""
    h = hands.reshape(len(hands), 3, 13).astype(np.float64)
    per_rank = h.sum(axis=1)                                          # (N,13)
    group = np.where(per_rank >= 3, 1.0, np.where(per_rank == 2, 0.5, 0.0))
    all_trash = (h[:, TREASURE].sum(1) + h[:, NEUTRAL].sum(1)) == 0
    trash_mult = np.where(all_trash, 2.0, 0.0)[:, None]               # 1 + (+1) or 1 + (-1)
    mult_cards = h[:, TREASURE] * 2.0 + h[:, NEUTRAL] * 1.0 + h[:, TRASH] * trash_mult
    return (BASE * (mult_cards + group * per_rank)).sum(1)


def add_matrix(k):
    """All multisets of k card types, as an (M, 39) count matrix."""
    combos = np.array(list(combinations_with_replacement(range(NTYPES), k)), dtype=np.int64)
    m = np.zeros((len(combos), NTYPES), dtype=np.int16)
    for col in range(k):
        np.add.at(m, (np.arange(len(combos)), combos[:, col]), 1)
    return m


CARD_COEF = np.concatenate([BASE * 2.0, BASE * 1.0, BASE * 0.0])     # per-type value excl. group bonus
TRASH_FLIP = np.concatenate([np.zeros(26), BASE * 2.0])              # extra if the hand is all trash
GROUP = np.array([n * (1.0 if n >= 3 else 0.5 if n == 2 else 0.0) for n in range(16)])


def fast_scores(adds, kept):
    """Same as score(adds + kept), split so that the add-only parts are precomputed once.
    adds: dict of precomputed arrays, kept: (39,) counts."""
    pr_kept = kept.reshape(3, 13).sum(0)
    sc = adds["lin"] + kept @ CARD_COEF
    for r in range(13):                                               # group bonus via lookup
        sc += (BASE[r] * GROUP[pr_kept[r]:pr_kept[r] + adds["kmax"] + 1])[adds["per_rank"][:, r]]
    if kept[:26].sum() == 0:                                          # all-trash rule
        mask = adds["nontrash"] == 0
        sc[mask] += adds["flip"][mask] + kept @ TRASH_FLIP
    return sc


def best_hands(hand_types, k, avail):
    """Top-N sets of k new cards, each paired with its best discard.

    Every set of new cards appears at most once: for each set we keep only the
    discard that gives the highest final score. Returns [(score, final, added)]."""
    adds = add_matrix(k)
    adds = adds[(adds <= avail).all(axis=1)]                          # respect remaining pool
    pre = {
        "lin": adds @ CARD_COEF,
        "per_rank": adds.reshape(len(adds), 3, 13).sum(1).astype(np.intp),
        "nontrash": adds[:, :26].sum(1),
        "flip": adds @ TRASH_FLIP,
        "kmax": k,
    }
    hand_vec = np.bincount(hand_types, minlength=NTYPES)
    discards = sorted(set(combinations(sorted(hand_types), k)))       # distinct discards only
    best_sc = np.full(len(adds), -np.inf)                             # best score per new-card set
    best_d = np.zeros(len(adds), dtype=np.intp)                       # ... and which discard gave it
    for d, discard in enumerate(discards):
        kept = hand_vec - np.bincount(discard, minlength=NTYPES)
        sc = fast_scores(pre, kept)
        better = sc > best_sc
        best_sc[better] = sc[better]
        best_d[better] = d
    n = min(TOP_N, len(adds))
    top = np.argpartition(-best_sc, n - 1)[:n]
    top = top[np.argsort(-best_sc[top], kind="stable")]
    out = []
    for i in top:
        kept = hand_vec - np.bincount(discards[best_d[i]], minlength=NTYPES)
        out.append((best_sc[i], adds[i] + kept, adds[i].copy()))
    return out


def describe(final, added, hand, ttype, treasure, trash, neutral):
    """Render a final hand: kept original cards with real suits, new cards marked '+'."""
    kept_left = final - added
    kept, removed, new = [], [], []
    for c in hand:
        t = ttype(c)
        if kept_left[t] > 0:
            kept_left[t] -= 1
            kept.append(repr(c))
        else:
            removed.append(repr(c))
    suit_of = {TREASURE: treasure, TRASH: trash, NEUTRAL: f"[{'|'.join(neutral)}]"}
    for t in np.nonzero(added)[0]:
        cat, r = divmod(int(t), 13)
        new += [f"{rank_name(r + 2)}{suit_of[cat]}"] * int(added[t])
    return f"+ {' '.join(new):<16} - {' '.join(removed):<16} | keep {' '.join(kept)}"


def main():
    if len(sys.argv) >= 3:
        hand_s, treasure = sys.argv[1], sys.argv[2].upper()
    else:
        hand_s = input("9-card hand (e.g. 2H3H5S7C9DJHQSKDAC): ").strip()
        treasure = input("Treasure suit (S/H/D/C): ").strip().upper()
    if treasure not in SUITS:
        raise SystemExit("treasure suit must be one of S H D C")
    hand = parse_cards(hand_s)
    if len(hand) != 9:
        raise SystemExit(f"need exactly 9 cards, got {len(hand)}")
    if any(c.is_joker for c in hand):
        raise SystemExit("Treasures & Trash is played without jokers in the deck")

    trash, neutral = suit_roles(treasure)
    cat_of = {treasure: TREASURE, trash: TRASH, neutral[0]: NEUTRAL, neutral[1]: NEUTRAL}
    ttype = lambda c: cat_of[c.suit] * 13 + (c.rank - 2)
    hand_types = [ttype(c) for c in hand]
    hand_vec = np.bincount(hand_types, minlength=NTYPES)
    avail = pool_counts() - hand_vec
    if (avail < 0).any():
        raise SystemExit("hand holds more copies of a card than exist in the pool")

    print(f"Hand: {' '.join(map(repr, hand))}")
    print(f"Treasure: {treasure}   Trash: {trash}   Neutral: {'/'.join(neutral)}")
    print(f"Current score: {score(hand_vec[None])[0]:g}\n")
    for k in range(1, 4):
        print(f"=== K = {k} ===")
        for i, (sc, final, added) in enumerate(best_hands(hand_types, k, avail), 1):
            print(f"{i:>2}. {sc:>6g}  {describe(final, added, hand, ttype, treasure, trash, neutral)}")
        print()


if __name__ == "__main__":
    main()
