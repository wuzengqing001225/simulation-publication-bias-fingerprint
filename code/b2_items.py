"""B2 item bank: 22 self-generated constructs covering all 20 B0 construct pairs.

Item provenance
---------------
* IPIP items (openness, conscientiousness, agency/assertiveness) are public domain
  and used near-verbatim.
* Numeracy items follow the Lipkus/Schwartz expanded-numeracy format.
* CRT-2 items follow Thomson & Oppenheimer (2016).
* Myside-bias items follow the Stanovich & West Ford-Explorer / German-car pair.
* Linda: Tversky & Kahneman (1983) conjunction item.
* Denominator neglect: Kirkpatrick & Epstein ratio-bias trays.
* Proportion dominance: Fetherstonhaugh et al. / Bartels lives-saved framing.
* Brand Engagement in Self-Concept, overplacement, vaccine acceptance, policy
  support, perceived public support, hazard risk/benefit and purchase intent are
  written here as functionally equivalent paraphrases in the style of the standard
  instrument (the Twin-2K-500 wave-4 item texts were not carried into B0's
  artifacts, and the published BES/ risk-benefit items are copyrighted).
  This is declared as a deviation in the B2 memo.

Response formats
----------------
likert7 : 1-7 rating, `reverse=True` -> 8-v
pct     : 0-100 percentage, `reverse=True` -> 100-v
num     : open numeric, scored 1 if within `tol` of `key`
mc      : single letter, scored 1 if == key
choice  : single letter, scored 1 if == key (optimal option)
"""

LIK = "Rate 1-7 (1 = strongly disagree, 7 = strongly agree)."
POL = "Rate 1-7 (1 = strongly oppose, 7 = strongly support)."
LIB = "Rate 1-7 (1 = very conservative, 4 = moderate, 7 = very liberal)."
BUY = "Rate 1-7 (1 = definitely would not buy, 7 = definitely would buy)."
CMP = "Rate 1-7 (1 = much worse than average, 4 = about average, 7 = much better than average)."

def I(iid, construct, text, fmt, anchors=None, reverse=False, key=None, tol=None):
    return dict(iid=iid, construct=construct, text=text, fmt=fmt, anchors=anchors,
                reverse=reverse, key=key, tol=tol)

POLICIES = [
    ("raising the federal minimum wage to $15 an hour", False),
    ("a government-funded health insurance plan that covers every resident", False),
    ("stricter federal limits on carbon emissions from power plants", False),
    ("raising income taxes on households earning more than $400,000 a year", False),
    ("creating a path to citizenship for undocumented immigrants already in the country", False),
    ("requiring background checks on all gun purchases, including private sales", False),
    ("federal funding to make public preschool free for all four-year-olds", False),
    ("increasing federal spending on wind and solar energy development", False),
    ("increasing the federal military budget", True),
    ("cutting the federal corporate income tax rate", True),
]

HAZARDS = ["riding a bicycle in city traffic", "drinking alcoholic beverages",
           "living near a chemical manufacturing plant", "agricultural pesticide use"]

CATEGORIES = ["a new pair of running shoes", "a streaming video subscription",
              "organic fresh produce", "a name-brand smartphone", "store-brand cereal",
              "a household air purifier", "an electric or hybrid car", "bottled water",
              "a fitness-tracking wearable", "vitamins or dietary supplements",
              "a meal-delivery service", "second-hand clothing"]

def build_bank():
    B = []
    n = 0
    def add(*args, **kw):
        nonlocal n
        n += 1
        B.append(I(n, *args, **kw))

    # S1 political liberalism (3)
    for lab in ["your political views in general",
                "your views on social issues such as abortion and LGBT rights",
                "your views on economic issues such as taxes and government spending"]:
        add("P_political_liberalism", f"Where would you place {lab}?", "likert7", LIB)

    # S2 policy support (10)
    for txt, consv in POLICIES:
        add("W_policy_support_liberal", f"How much do you support or oppose {txt}?",
            "likert7", POL, reverse=consv)

    # S3 perceived public support (10)
    for txt, consv in POLICIES:
        add("W_perceived_public_support_lib",
            f"Out of every 100 American adults, how many do you think support {txt}? "
            "Answer with a number from 0 to 100.", "pct", None, reverse=consv)

    # S4 personality, interleaved (12)
    P4 = [("P_openness", "I have a vivid imagination.", False),
          ("P_conscientiousness", "I am always prepared.", False),
          ("P_agency", "I take charge of situations.", False),
          ("P_openness", "I have difficulty understanding abstract ideas.", True),
          ("P_conscientiousness", "I leave my belongings around.", True),
          ("P_agency", "I wait for others to lead the way.", True),
          ("P_openness", "I enjoy hearing new and different ideas.", False),
          ("P_conscientiousness", "I pay attention to details.", False),
          ("P_agency", "I try to lead others.", False),
          ("P_openness", "I am not interested in abstract ideas.", True),
          ("P_conscientiousness", "I shirk my duties.", True),
          ("P_agency", "I take control of things.", False)]
    for c, t, rev in P4:
        add(c, t, "likert7", LIK, reverse=rev)

    # S5 hazard benefit then risk (8)
    for h in HAZARDS:
        add("W_benefit_perception", f"How much benefit to society is there in {h}?",
            "likert7", "Rate 1-7 (1 = no benefit at all, 7 = very great benefit).")
    for h in HAZARDS:
        add("W_risk_perception", f"How much risk to people's health and safety is there in {h}?",
            "likert7", "Rate 1-7 (1 = no risk at all, 7 = very great risk).")

    # S6 myside: Ford (1)
    add("W_myside_ford",
        "The German government has determined that a sport-utility vehicle made by an American "
        "company, the Ford Explorer, has an accident rate roughly eight times higher than the "
        "average German vehicle, and is considering banning it from German roads. Should Germany "
        "be allowed to ban this American-made vehicle?",
        "likert7", "Rate 1-7 (1 = definitely should not be allowed, 7 = definitely should be allowed).")

    # S7 numeracy (4)
    add("P_numeracy", "Imagine a fair coin is flipped 1,000 times. Out of 1,000 flips, how many "
        "times would you expect the coin to come up heads?", "num", "Answer with a number.",
        key=500, tol=0.5)
    add("P_numeracy", "In a lottery, the chance of winning a $10 prize is 1%. If 1,000 people each "
        "buy one ticket, how many of them would you expect to win a $10 prize?", "num",
        "Answer with a number.", key=10, tol=0.5)
    add("P_numeracy", "In a sweepstakes, the chance of winning a car is 1 in 1,000. What percentage "
        "of sweepstakes tickets win a car?", "num", "Answer with a number (percent).",
        key=0.1, tol=0.005)
    add("P_numeracy", "If the chance of getting a disease is 20 out of 100, this is the same as a "
        "___ percent chance of getting the disease.", "num", "Answer with a number (percent).",
        key=20, tol=0.5)

    # S8 CRT-2 (4)
    add("P_crt2_score", "The mother of Sarah has four daughters: Spring, Summer, Autumn and who "
        "else?", "mc", "Answer A, B, C or D.  A) Winter  B) Sarah  C) Rain  D) Sunny", key="B")
    add("P_crt2_score", "How many cubic feet of dirt are in a hole that is 3 feet deep, 3 feet wide "
        "and 3 feet long?", "num", "Answer with a number.", key=0, tol=0.001)
    add("P_crt2_score", "If you are running a race and you pass the second-place runner, what place "
        "are you in?", "mc", "Answer A, B, C or D.  A) First  B) Second  C) Third  D) Cannot be "
        "determined", key="B")
    add("P_crt2_score", "A farmer had 15 sheep and all but 8 died. How many sheep are left alive?",
        "num", "Answer with a number.", key=8, tol=0.001)

    # S9 verbal / reasoning composite proxy (5)
    add("P_actual_total", "Which word is closest in meaning to ABSTRUSE?", "mc",
        "Answer A, B, C or D.  A) obscure  B) rough  C) hollow  D) sudden", key="A")
    add("P_actual_total", "Which word is closest in meaning to PLACATE?", "mc",
        "Answer A, B, C or D.  A) provoke  B) appease  C) postpone  D) misjudge", key="B")
    add("P_actual_total", "Which word is most nearly OPPOSITE in meaning to PROLIFIC?", "mc",
        "Answer A, B, C or D.  A) abundant  B) barren  C) rapid  D) fertile", key="B")
    add("P_actual_total", "Which word is most nearly OPPOSITE in meaning to MITIGATE?", "mc",
        "Answer A, B, C or D.  A) soften  B) aggravate  C) clarify  D) delay", key="B")
    add("P_actual_total", "What number comes next in the series 2, 6, 12, 20, 30, ___ ?", "num",
        "Answer with a number.", key=42, tol=0.001)

    # S10 probability-maximizing card task (10)
    for t in range(1, 11):
        add("W_prob_maximizing_cards",
            f"Trial {t} of a card game. A shuffled deck contains 70 RED cards and 30 BLUE cards. "
            "One card will be drawn at random and you must guess its colour before it is revealed; "
            "you earn $1 for each correct guess. What is your guess for this trial?",
            "choice", "Answer R or B.", key="R")

    # S11 denominator neglect / ratio bias (3)
    add("W_denom_neglect_avoid",
        "Two trays of jelly beans are offered. Tray A holds 10 beans, 1 of which is a winner. "
        "Tray B holds 100 beans, 9 of which are winners. You draw one bean without looking and win "
        "a prize only if it is a winner. Which tray do you draw from?", "choice",
        "Answer A or B.", key="A")
    add("W_denom_neglect_avoid",
        "Two bowls of tickets are offered. Bowl A holds 200 tickets, 18 of which win. Bowl B holds "
        "20 tickets, 2 of which win. You draw one ticket at random. Which bowl do you draw from?",
        "choice", "Answer A or B.", key="B")
    add("W_denom_neglect_avoid",
        "Two boxes of marbles are offered. Box A holds 30 marbles, 3 of which win. Box B holds 300 "
        "marbles, 28 of which win. You draw one marble at random. Which box do you draw from?",
        "choice", "Answer A or B.", key="A")

    # S12 Linda conjunction (2)
    add("W_linda_conjunction_rating",
        "Linda is 31, single, outspoken and very bright. She majored in philosophy. As a student she "
        "was deeply concerned with issues of discrimination and social justice, and took part in "
        "anti-nuclear demonstrations. How likely is it that Linda is a bank teller AND is active in "
        "the feminist movement?", "likert7",
        "Rate 1-7 (1 = extremely unlikely, 7 = extremely likely).")
    add("W_linda_conjunction_rating",
        "Robert is 45, quiet and methodical, and spends his weekends restoring antique clocks. He "
        "studied mechanical engineering. How likely is it that Robert is an insurance salesman AND "
        "a member of a local historical society?", "likert7",
        "Rate 1-7 (1 = extremely unlikely, 7 = extremely likely).")

    # S13 proportion dominance (2)
    add("W_proportion_dom_1A",
        "A public health agency must fund one of two flood-safety programs. Program A would save "
        "1,000 lives out of the 10,000 people at risk in one region. Program B would save 2,000 "
        "lives out of the 100,000 people at risk in a different region. How strongly do you favour "
        "funding Program A (the one that saves the larger share of those at risk)?", "likert7",
        "Rate 1-7 (1 = strongly favour Program B, 4 = no preference, 7 = strongly favour Program A).")
    add("W_proportion_dom_1A",
        "A relief fund must choose one of two airport-safety upgrades. Upgrade A would save 150 of "
        "the 200 passengers at risk on one route. Upgrade B would save 300 of the 3,000 passengers "
        "at risk across several routes. How strongly do you favour funding Upgrade A (the one that "
        "saves the larger share of those at risk)?", "likert7",
        "Rate 1-7 (1 = strongly favour Upgrade B, 4 = no preference, 7 = strongly favour Upgrade A).")

    # S14 purchase intent (12)
    for c in CATEGORIES:
        add("W_purchase_intent", f"How likely are you to buy {c} in the next three months?",
            "likert7", BUY)

    # S15 brand engagement in self-concept (4)
    for t in ["I feel a special connection to the brands I prefer.",
              "The brands I use say something important about who I am.",
              "I do not think of any brand as part of my identity.",
              "Some of the brands in my life matter to me personally."]:
        add("P_BES", t, "likert7", LIK, reverse=(t.startswith("I do not think")))

    # S16 overplacement (3)
    for lab in ["your ability to drive safely", "your general intelligence",
                "your ability to get along with other people"]:
        add("P_overplacement", f"Compared with the average person of your age, how would you rate "
            f"{lab}?", "likert7", CMP)

    # S17 religiosity -> nonattendance (2)
    add("P_religious_nonattendance", "How often do you attend religious services?", "likert7",
        "Rate 1-7 (1 = never, 4 = a few times a year, 7 = more than once a week).", reverse=True)
    add("P_religious_nonattendance", "Religion is an important part of my daily life.", "likert7",
        LIK, reverse=True)

    # S18 vaccine acceptance (3)
    add("W_vaccine_acceptance", "I would get a seasonal flu vaccine this year if it were offered to "
        "me for free.", "likert7", LIK)
    add("W_vaccine_acceptance", "I would accept an updated COVID-19 booster shot if my doctor "
        "recommended it.", "likert7", LIK)
    add("W_vaccine_acceptance", "Vaccines approved by public health authorities are safe and "
        "effective.", "likert7", LIK)

    # S19 myside: German car (1)
    add("W_myside_german",
        "The United States government has determined that a sport-utility vehicle made by a German "
        "company has an accident rate roughly eight times higher than the average American vehicle, "
        "and is considering banning it from American roads. Should the United States be allowed to "
        "ban this German-made vehicle?",
        "likert7", "Rate 1-7 (1 = definitely should not be allowed, 7 = definitely should be allowed).")

    return B


BANK = build_bank()
CONSTRUCTS = sorted({b["construct"] for b in BANK})
