# Copyright 2026 Mathieu Fourment
# SPDX-License-Identifier: MIT
"""A synthetic population of people known under several names.

Each person has a main identity (first name, surname, date of birth, postcode) and
may have others: the nicknames their first name carries, a second surname (a
maiden or married name), a second date of birth (a day and month swapped, or a
year off by one, which is how a wrong date usually enters a register). A record
is one sighting of a person: it presents *one* of those identities in its main
columns and lists *some* of the others in three alias columns, because the source
that wrote it knew only part of the story. Typos and missing values land on top.

So two records of one person may agree outright, may agree only because one's
main name is in the other's alias list, or may share nothing but a postcode. And
two different people may share a nickname (`ted` is both an `edward` and a
`theodore`), which is the case where membership is right and intersection of
the alias lists is wrong.

Writes into the output directory:

  alias.parquet        unique_id, first_name, surname, dob, postcode and the
                       three alias lists; no entity column
  alias.entities.csv   unique_id,entity, for the scorer
  alias.truth.csv      id_a,id_b, every within-entity pair, for `recall`

Everything is drawn from one seed, so a run is reproducible to the byte.
"""

import argparse
import csv
import datetime
import itertools
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Real English nicknames, including the collisions that make the problem hard:
# `ted`, `kit`, `sandy`, `pat` and `al` each belong to more than one name.
NICKNAMES = {
    "william": ["bill", "will", "billy", "liam"],
    "robert": ["bob", "rob", "bobby", "bert"],
    "richard": ["dick", "rick", "rich"],
    "elizabeth": ["liz", "beth", "betty", "eliza"],
    "margaret": ["maggie", "peggy", "meg"],
    "katherine": ["kate", "kathy", "kit"],
    "james": ["jim", "jimmy", "jamie"],
    "john": ["jack", "johnny"],
    "edward": ["ed", "ted", "eddie"],
    "theodore": ["ted", "theo"],
    "thomas": ["tom", "tommy"],
    "charles": ["charlie", "chuck"],
    "michael": ["mike", "mick"],
    "anthony": ["tony"],
    "joseph": ["joe", "joey"],
    "daniel": ["dan", "danny"],
    "alexander": ["alex", "sandy", "al"],
    "alexandra": ["alex", "sandy", "lexi"],
    "albert": ["al", "bert"],
    "christopher": ["chris", "kit"],
    "christine": ["chris", "tina"],
    "patricia": ["pat", "patty", "trish"],
    "patrick": ["pat", "paddy"],
    "susan": ["sue", "susie"],
    "jennifer": ["jen", "jenny"],
    "deborah": ["deb", "debbie"],
    "rebecca": ["becky", "becca"],
    "victoria": ["vicky", "tori"],
    "samuel": ["sam", "sammy"],
    "samantha": ["sam", "sammy"],
    "benjamin": ["ben", "benny"],
    "nicholas": ["nick", "nicky"],
    "matthew": ["matt"],
    "andrew": ["andy", "drew"],
    "jonathan": ["jon", "jonny"],
    "frederick": ["fred", "freddie"],
    "dorothy": ["dot", "dolly"],
    "barbara": ["barb", "babs"],
    "henry": ["harry", "hank"],
    "catherine": ["cathy", "kate"],
}

CONSONANTS = "bcdfghjklmnprstvwz"
VOWELS = "aeiou"


def zipf_weights(count, exponent):
    weights = 1.0 / np.arange(1, count + 1) ** exponent
    return weights / weights.sum()


def made_up_names(rng, count, syllables):
    """Pronounceable strings, distinct, so the pool has the size it claims."""
    seen = set()
    while len(seen) < count:
        length = rng.integers(syllables[0], syllables[1] + 1)
        seen.add(
            "".join(
                rng.choice(list(CONSONANTS)) + rng.choice(list(VOWELS))
                for _ in range(length)
            )
            + (rng.choice(list(CONSONANTS)) if rng.random() < 0.5 else "")
        )
    names = sorted(seen)
    rng.shuffle(names)
    return names


def typo(rng, value):
    """One edit: a substitution, deletion, insertion or adjacent transposition."""
    if len(value) < 2:
        return value
    i = int(rng.integers(0, len(value) - 1))
    kind = rng.integers(0, 4)
    letter = rng.choice(list("abcdefghijklmnopqrstuvwxyz"))
    if kind == 0:
        return value[:i] + letter + value[i + 1 :]
    if kind == 1:
        return value[:i] + value[i + 1 :]
    if kind == 2:
        return value[:i] + letter + value[i:]
    return value[:i] + value[i + 1] + value[i] + value[i + 2 :]


def wrong_date(rng, date):
    """The second date of birth: day and month swapped where that is a date,
    otherwise the year off by one."""
    if date.day <= 12 and date.day != date.month and rng.random() < 0.6:
        return date.replace(month=date.day, day=date.month)
    year = date.year + (1 if rng.random() < 0.5 else -1)
    try:
        return date.replace(year=year)
    except ValueError:  # the 29th of February
        return date.replace(year=year, day=28)


def make_people(rng, count, args):
    canonical = sorted(NICKNAMES)
    plain_first = made_up_names(rng, args.first_names, (2, 3))
    surnames = made_up_names(rng, args.surnames, (2, 4))
    postcodes = [
        f"{a}{b}{n} {m}{c}{d}"
        for a, b, n, m, c, d in zip(
            rng.choice(list("ABCDEFGHKLMNPRS"), args.postcodes),
            rng.choice(list("ABCDEFGHKLMNPRS"), args.postcodes),
            rng.integers(1, 30, args.postcodes),
            rng.integers(1, 10, args.postcodes),
            rng.choice(list("ABDEFGHJLNPQRSTUWXYZ"), args.postcodes),
            rng.choice(list("ABDEFGHJLNPQRSTUWXYZ"), args.postcodes),
            strict=True,
        )
    ]
    first_weights = zipf_weights(len(plain_first), 1.0)
    canonical_weights = zipf_weights(len(canonical), 0.6)
    surname_weights = zipf_weights(len(surnames), args.surname_zipf)
    start = datetime.date(1930, 1, 1).toordinal()
    span = datetime.date(2005, 12, 31).toordinal() - start

    # Drawn up front: a weighted choice per person would rebuild the cumulative
    # weights every time.
    is_canonical = rng.random(count) < args.canonical_share
    canonical_draw = rng.choice(len(canonical), count, p=canonical_weights)
    plain_draw = rng.choice(len(plain_first), count, p=first_weights)
    surname_draw = rng.choice(len(surnames), (count, 2), p=surname_weights)

    people = []
    for i in range(count):
        if is_canonical[i]:
            first = canonical[canonical_draw[i]]
        else:
            first = plain_first[plain_draw[i]]
        surname = surnames[surname_draw[i, 0]]
        dob = datetime.date.fromordinal(start + int(rng.integers(0, span)))
        person = {
            "first": [first, *NICKNAMES.get(first, [])],
            "surname": [surname],
            "dob": [dob.isoformat()],
            "postcode": postcodes[int(rng.integers(0, len(postcodes)))],
        }
        if rng.random() < args.alt_surname:
            other = surnames[surname_draw[i, 1]]
            if other != surname:
                person["surname"].append(other)
        if rng.random() < args.alt_dob:
            other = wrong_date(rng, dob).isoformat()
            if other != person["dob"][0]:
                person["dob"].append(other)
        people.append(person)
    return people, postcodes


def sighting(rng, person, postcodes, args):
    """One record: which identity it presents, which of the others it lists."""

    def present(names, other_share):
        if len(names) > 1 and rng.random() < other_share:
            return int(rng.integers(1, len(names)))
        return 0

    knows_aliases = rng.random() < args.alias_coverage
    record = {}
    for field, other_share in (
        ("first", args.use_nickname),
        ("surname", args.use_alt_surname),
        ("dob", args.use_alt_dob),
    ):
        names = person[field]
        shown = present(names, other_share)
        value = names[shown]
        listed = []
        if knows_aliases:
            listed = [
                name
                for i, name in enumerate(names)
                if i != shown and rng.random() < args.alias_recall
            ]
        if field != "dob" and rng.random() < args.typo:
            value = typo(rng, value)
        record[field] = value
        record[field + "_aliases"] = sorted(set(listed) - {value})
    postcode = person["postcode"]
    if rng.random() < args.moved:
        postcode = postcodes[int(rng.integers(0, len(postcodes)))]
    record["postcode"] = postcode
    for field, rate in (
        ("first", args.null_first),
        ("surname", args.null_surname),
        ("dob", args.null_dob),
        ("postcode", args.null_postcode),
    ):
        if rng.random() < rate:
            record[field] = None
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="directory to write into")
    parser.add_argument("--people", type=int, default=60000)
    parser.add_argument("--seed", type=int, default=20261119)
    parser.add_argument("--first-names", type=int, default=3000)
    parser.add_argument("--surnames", type=int, default=8000)
    parser.add_argument("--surname-zipf", type=float, default=1.05)
    parser.add_argument("--postcodes", type=int, default=4000)
    parser.add_argument("--canonical-share", type=float, default=0.45)
    parser.add_argument("--alt-surname", type=float, default=0.3)
    parser.add_argument("--alt-dob", type=float, default=0.1)
    parser.add_argument(
        "--duplicate-rate",
        type=float,
        default=0.8,
        help="mean extra records per person (Poisson)",
    )
    parser.add_argument("--use-nickname", type=float, default=0.4)
    parser.add_argument("--use-alt-surname", type=float, default=0.5)
    parser.add_argument("--use-alt-dob", type=float, default=0.5)
    parser.add_argument(
        "--alias-coverage",
        type=float,
        default=0.7,
        help="share of records whose source lists any aliases",
    )
    parser.add_argument(
        "--alias-recall",
        type=float,
        default=0.6,
        help="share of a person's other names such a source lists",
    )
    parser.add_argument("--typo", type=float, default=0.05)
    parser.add_argument("--moved", type=float, default=0.2)
    parser.add_argument("--null-first", type=float, default=0.02)
    parser.add_argument("--null-surname", type=float, default=0.01)
    parser.add_argument("--null-dob", type=float, default=0.03)
    parser.add_argument("--null-postcode", type=float, default=0.1)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    people, postcodes = make_people(rng, args.people, args)
    rows = []
    entities = []
    for index, person in enumerate(people):
        copies = 1 + int(rng.poisson(args.duplicate_rate))
        for copy in range(copies):
            record = sighting(rng, person, postcodes, args)
            record["unique_id"] = f"p{index}-{copy}"
            rows.append(record)
            entities.append((record["unique_id"], f"p{index}"))
    order = rng.permutation(len(rows))
    rows = [rows[i] for i in order]
    entities = [entities[i] for i in order]

    os.makedirs(args.out, exist_ok=True)
    string = pa.string()
    strings = pa.list_(pa.string())
    table = pa.table(
        {
            "unique_id": pa.array([r["unique_id"] for r in rows], string),
            "first_name": pa.array([r["first"] for r in rows], string),
            "surname": pa.array([r["surname"] for r in rows], string),
            "dob": pa.array([r["dob"] for r in rows], string),
            "postcode": pa.array([r["postcode"] for r in rows], string),
            "first_name_aliases": pa.array([r["first_aliases"] for r in rows], strings),
            "surname_aliases": pa.array([r["surname_aliases"] for r in rows], strings),
            "dob_aliases": pa.array([r["dob_aliases"] for r in rows], strings),
        }
    )
    pq.write_table(table, os.path.join(args.out, "alias.parquet"))

    with open(os.path.join(args.out, "alias.entities.csv"), "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["unique_id", "entity"])
        writer.writerows(entities)
    members = {}
    for unique_id, entity in entities:
        members.setdefault(entity, []).append(unique_id)
    with open(os.path.join(args.out, "alias.truth.csv"), "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id_a", "id_b"])
        for ids in members.values():
            writer.writerows(itertools.combinations(ids, 2))
    print(
        f"{len(rows):,} records of {args.people:,} people, "
        f"{sum(len(m) * (len(m) - 1) // 2 for m in members.values()):,} true pairs"
    )


if __name__ == "__main__":
    main()
