"""ceil2x2 item bank: 14 new constructs for the 2x2 ceiling experiment.

Companion to b2_items.py, which is the format authority. Only the 14
constructs that b2_items.py does not already cover are defined here; the
other 9 constructs of the ceil2x2 pair design are imported from B2.

Item provenance (parallel forms, not verbatim copies)
-----------------------------------------------------
Every battery below is written to match the *operationalization style* of the
standard instrument used for the same-named construct in the Twin-2K-500
panel -- same facet coverage, same response format, same keying balance --
but with wording composed here. Per-construct benchmarks are noted inline
and collected in ceil2x2_items_notes.md.

Response formats
----------------
likert7 : 1-7 rating, `reverse=True` -> 8-v
pct     : 0-100 percentage, `reverse=True` -> 100-v
num     : open numeric, scored 1 if within `tol` of `key`; when `key is None`
          the raw number is the score (e.g. dollars given, dollars offered)
mc      : single letter, scored 1 if == key
choice  : single letter, scored 1 if == key (optimal option)
cat     : self-reported category, answered as the bracket number listed in
          `anchors`; scored as that bracket ordinal (monotone in age /
          attainment / income). No `key` -- these are not ability items.
"""

LIK = "Rate 1-7 (1 = strongly disagree, 7 = strongly agree)."
WLD = "Rate 1-7 (1 = definitely would not, 7 = definitely would)."
TF = "Answer A or B.  A) True  B) False"

AGE = ("Answer with the number of the bracket that applies.  1) 18-24  2) 25-34  "
       "3) 35-44  4) 45-54  5) 55-64  6) 65 or older")
EDU = ("Answer with the number of the highest level you have completed.  "
       "1) Less than high school  2) High school diploma or GED  "
       "3) Some college, no degree  4) Associate degree  5) Bachelor's degree  "
       "6) Master's degree  7) Doctoral or professional degree")
INC = ("Answer with the number of the bracket that applies to your household's total "
       "income last year, before taxes.  1) Under $25,000  2) $25,000-$49,999  "
       "3) $50,000-$74,999  4) $75,000-$99,999  5) $100,000-$149,999  "
       "6) $150,000-$199,999  7) $200,000 or more")


def I(iid, construct, text, fmt, anchors=None, reverse=False, key=None, tol=None):
    return dict(iid=iid, construct=construct, text=text, fmt=fmt, anchors=anchors,
                reverse=reverse, key=key, tol=tol)


# --- P_RFS: Altemeyer & Hunsberger 12-item Revised Religious Fundamentalism
# short form style -- balanced 6 pro / 6 con, generic-monotheistic wording,
# themes of one true faith, inerrant scripture, cosmic evil, no compromise.
RFS = [
    ("God has given humanity one complete and unerring guide to right belief and right "
     "living.", False),
    ("The core teachings of the one true faith must never be adjusted to suit modern "
     "opinion.", False),
    ("Anyone who sincerely follows the one true religion is protected from serious "
     "spiritual error.", False),
    ("There is a real struggle in the world between the followers of God and the forces "
     "of evil.", False),
    ("Sacred scripture, properly understood, contains no factual mistakes of any kind.", False),
    ("In the end, salvation is open only to those who accept the one true faith.", False),
    ("No single religion can claim to hold the whole truth about God.", True),
    ("Sacred texts are human documents and should be read as products of their own time.", True),
    ("Religious teaching should be revised when it conflicts with well-established "
     "knowledge.", True),
    ("Good and evil are matters of human choice, not a battle between cosmic forces.", True),
    ("People of many different faiths, and of no faith at all, can lead equally moral "
     "lives.", True),
    ("Questioning and doubt make a person's faith stronger rather than weaker.", True),
]

# --- P_agreeableness: BFI-2 domain style, three facets (compassion, respectfulness,
# trust), 3 keyed-true / 3 keyed-false.
AGREEABLE = [
    ("I feel sympathy for people who are going through a hard time.", False),
    ("I give people the benefit of the doubt.", False),
    ("I stay courteous with others even when I disagree with them.", False),
    ("I am quick to point out other people's shortcomings.", True),
    ("I can be indifferent to how other people feel.", True),
    ("I assume most people will try to take advantage of me if they can.", True),
]

# --- P_neuroticism: BFI-2 domain style, three facets (anxiety, depression,
# emotional volatility), 4 keyed-true / 2 keyed-false.
NEUROTIC = [
    ("I worry about things that may never actually happen.", False),
    ("My mood can shift quickly for no clear reason.", False),
    ("I get discouraged easily when something goes wrong.", False),
    ("I often feel tense or on edge.", False),
    ("I stay calm when I am under pressure.", True),
    ("I rarely feel sad or low.", True),
]

# --- P_maximization: Schwartz et al. Maximization Scale style, three facets
# (search for alternatives, decision difficulty, high standards), 5 keyed-true /
# 1 keyed-false.
MAXIM = [
    ("When I shop, I want to see every option before I decide.", False),
    ("Even when I am fairly happy with a choice, I keep an eye out for something "
     "better.", False),
    ("Picking a gift for a friend is hard for me.", False),
    ("Choosing what to watch or listen to is difficult because there are so many "
     "options.", False),
    ("I often imagine a version of my life that is much better than the one I have.", False),
    ("I am content to take the first option that meets my needs.", True),
]

# --- P_fluid: number-series induction in the Raven/Cattell verbal-numeric idiom.
# Deliberately no matrix or figural items (text-only channel).
SERIES = [
    ("3, 6, 11, 18, 27", 38),
    ("2, 3, 5, 8, 13, 21", 34),
    ("96, 48, 24, 12", 6),
    ("5, 6, 9, 14, 21", 30),
    ("4, 7, 13, 25, 49", 97),
    ("1, 2, 6, 24, 120", 720),
]


def build_bank():
    B = []
    n = 0

    def add(*args, **kw):
        nonlocal n
        n += 1
        B.append(I(n, *args, **kw))

    # C1 religious fundamentalism (12)
    for t, rev in RFS:
        add("P_RFS", t, "likert7", LIK, reverse=rev)

    # C2 agreeableness (6)
    for t, rev in AGREEABLE:
        add("P_agreeableness", t, "likert7", LIK, reverse=rev)

    # C3 neuroticism (6)
    for t, rev in NEUROTIC:
        add("P_neuroticism", t, "likert7", LIK, reverse=rev)

    # C4 maximization (6)
    for t, rev in MAXIM:
        add("P_maximization", t, "likert7", LIK, reverse=rev)

    # C5 fluid reasoning (6)
    for s, k in SERIES:
        add("P_fluid", f"What number comes next in the series {s}, ___ ?", "num",
            "Answer with a number.", key=k, tol=0.001)

    # C6 financial literacy -- Lusardi & Mitchell "Big Three" style: compounding,
    # real interest / inflation, single-stock vs. mutual-fund diversification (3)
    add("P_finliteracy",
        "Suppose you put $200 in a savings account that pays 2% interest a year. You make "
        "no deposits or withdrawals. After 5 years, how much would be in the account?",
        "mc", "Answer A, B or C.  A) More than $220  B) Exactly $220  C) Less than $220",
        key="A")
    add("P_finliteracy",
        "Suppose your savings account pays 1% interest a year while prices rise by 2% a "
        "year. After one year, how much could you buy with the money in that account?",
        "mc", "Answer A, B or C.  A) More than today  B) Exactly the same as today  "
        "C) Less than today", key="C")
    add("P_finliteracy",
        "Putting your money into the shares of one single company is usually safer than "
        "putting it into a stock mutual fund that holds many companies.", "mc", TF, key="B")

    # C7 Wason selection task, text-rendered -- abstract letter/number version plus a
    # thematic version, four-option forced choice (2)
    add("P_wason",
        "Four cards lie on a table. Each has a letter on one face and a number on the "
        "other. The faces you can see read: A, K, 4, 7. Someone claims: \"If a card has a "
        "vowel on its letter face, then it has an even number on its number face.\" Which "
        "cards must you turn over -- and only those -- to find out whether the claim is "
        "false?", "mc",
        "Answer A, B, C or D.  A) A and 4  B) A only  C) A and 7  D) A, 4 and 7", key="C")
    add("P_wason",
        "A shop's catalogue makes this claim: \"If a book is a hardcover, then it costs "
        "more than $30.\" Four books sit on a shelf. For two of them you can see only the "
        "binding (one hardcover, one paperback); for the other two you can see only the "
        "price ($45 and $19). Which books must you check -- and only those -- to find out "
        "whether the claim is false?", "mc",
        "Answer A, B, C or D.  A) the hardcover and the $45 book  B) the hardcover and the "
        "$19 book  C) the hardcover only  D) the hardcover, the $45 book and the $19 book",
        key="B")

    # C8 dictator game, sender side -- one-shot anonymous allocation of a $10
    # endowment (Forsythe/Hoffman dictator paradigm) (1)
    add("P_dictator_sender",
        "You have been given $10. You are paired at random with another participant who "
        "has been given nothing and who will never learn who you are. You decide how much "
        "of the $10, if any, to pass to that person; you keep the rest. Your decision is "
        "final and anonymous. How many dollars do you pass to the other person?", "num",
        "Answer with a whole number of dollars from 0 to 10.")

    # C9 base-rate neglect, engineer/lawyer with a 30-engineer / 70-lawyer base rate
    # (Kahneman & Tversky 1973 lawyer-engineer paradigm; normative answer 30) (1)
    add("W_base_rate_30eng",
        "A panel of psychologists interviewed 100 professionals: 30 of them are engineers "
        "and 70 are lawyers. From those interviews they wrote short personality sketches. "
        "One sketch, drawn at random, reads: \"Jack is 45, married, with four children. He "
        "is conservative, careful and ambitious. He shows no interest in political or "
        "social issues and spends his free time on home carpentry, sailing and "
        "mathematical puzzles.\" What is the probability, in percent, that Jack is one of "
        "the 30 engineers?", "pct",
        "Answer with a number from 0 to 100 (percent chance).")

    # C10 less-is-more / evaluability, dinnerware set A judged in isolation
    # (Hsee 1998 joint-separate evaluation; A = 24 pieces, all intact) (1)
    add("W_less_is_more_A",
        "You are browsing a clearance table at a home-goods store and find one dinnerware "
        "set, sold as-is. The set contains 24 pieces: 8 dinner plates, 8 soup bowls and 8 "
        "dessert plates. Every piece is intact. The pattern and quality are ordinary. "
        "What is the most you would be willing to pay for this set?", "num",
        "Answer with a dollar amount.")

    # C11 mental accounting -- three Thaler-style topical-account scenarios: lost
    # theatre ticket, earmarked windfall, and a paid-in-advance sunk cost (3)
    add("P_mentalaccounting",
        "You bought a $60 ticket to a concert weeks ago. At the door you discover the "
        "ticket is gone and it cannot be replaced or refunded. Identical seats are still "
        "on sale at the box office for $60, and you can afford it. Would you buy another "
        "ticket and see the concert?", "likert7", WLD, reverse=True)
    add("P_mentalaccounting",
        "A relative gives you $500 and says it is \"for a trip somewhere nice.\" You put "
        "it in your account. Your car then needs a $500 repair that you would otherwise "
        "have to put on a credit card. Spending that particular $500 on the car repair "
        "instead of on a trip would feel wrong to me.", "likert7", LIK)
    add("P_mentalaccounting",
        "You paid $80 in advance for a full-day ski pass. The morning arrives cold and "
        "wet, the runs are icy, and you know you would not enjoy the day; the $80 is not "
        "refundable and you have no other plans. Would you go skiing anyway?", "likert7",
        WLD)

    # C12 age bracket -- standard panel demographic, scored as bracket ordinal (1)
    add("P_age_bracket", "Which age group are you in?", "cat", AGE)

    # C13 education -- standard panel demographic, scored as attainment ordinal (1)
    add("P_education", "What is the highest level of schooling you have completed?",
        "cat", EDU)

    # C14 household income -- standard panel demographic, scored as bracket ordinal (1)
    add("P_income", "Which bracket best describes your total household income last year, "
        "before taxes?", "cat", INC)

    return B


BANK = build_bank()
CONSTRUCTS = sorted({b["construct"] for b in BANK})
