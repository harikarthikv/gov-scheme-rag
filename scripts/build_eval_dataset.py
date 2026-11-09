"""Deterministically extract source-authored FAQ pairs; never call a generator.

Run from the project root: python -m scripts.build_eval_dataset.
The committed output is a reviewed fixture, not regenerated during evaluation.
"""

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

import pymupdf

from gov_scheme_rag.config import DATA_DIR, PROJECT_ROOT
from gov_scheme_rag.identity import canonical_filename, normalized_text

gold_standard = json.loads((PROJECT_ROOT / "evaluation/seeds.json").read_text(encoding="utf-8"))

QUESTION = re.compile(
    r"(?:What|Who|When|Where|Why|How|Can|Is|Are|Does|Do|Will|Which|Whether)\b[^?]{8,220}\?"
)

# Rejected during source review: mismatched scheme FAQ and clearly wrong FAQs.
EXCLUDED_TITLES = {"Financial Assistance For Widows, Divorced, Separated / Destitute Deserted Or Abandoned Women",
                   "Madilu Kit"}

# Written from the reviewed source answers before any retrieval measurement.
# Keep realistic paraphrases alongside exact FAQ questions, with the same qrels.
PARAPHRASES = {
    11: "Which Uttarakhand board runs the Child Benefit Scheme for construction workers?",
    12: "Under the UKBOCWWB Child Benefit Scheme, which stages of a child's education receive assistance?",
    13: "Give examples of traditional performances supported by Goa's scheme to preserve Maand culture.",
    14: "How does the Goan Maand preservation scheme encourage young people to learn traditional arts?",
    15: "What do Meghalaya's seed testing services help farmers check before sowing?",
    16: "Can a farmer based outside Meghalaya apply for its Seed Testing Laboratory Scheme?",
    17: "For the Goan diaspora IT lease rebate, how long must a unit have occupied its office before applying?",
    18: "What share of rent is reimbursed for the first six months under Goa's Lease Rental Rebate Scheme for Goan Diaspora?",
    21: "Why does Tamil Nadu give the Poompuhar State Award to craftspersons?",
    22: "What cash and non-cash rewards come with the Poompuhar State Award?",
    24: "How is the Gujarat construction workers' housing subsidy spread over five years?",
    26: "Does a small business have to obtain ZED certification, or is it optional?",
    27: "What does Jharkhand's coaching and allied scheme help Scheduled Caste candidates prepare for?",
    30: "What is the upper age cutoff for applicants to the National Awards to Teachers in Higher Education?",
    32: "Who is Chhattisgarh's silicosis financial and rehabilitation assistance intended to help?",
    36: "Can the Chandigarh construction workers' interest-free loan be used to buy a bicycle or a fridge?",
    39: "For how many years at most can an INSPIRE SHE scholar receive support?",
    40: "What must an INSPIRE SHE student do to keep receiving the scholarship?",
    41: "May I work part-time while receiving the ICAR post-matric scholarship for SC/ST students?",
    45: "After buying a vehicle, how do I add its details to the GST exemption certificate portal?",
    50: "Do I need registration with Assam's construction workers welfare board to get its medical assistance?",
    54: "What minimum disability percentage is required for Puducherry's capital investment subsidy for physically handicapped entrepreneurs?",
    56: "How long must a Gujarat worker have worked continuously before qualifying for the Labour Welfare Board maternity assistance?",
    63: "Can a previous Tenzing Norgay National Adventure Award winner receive the same award in another category?",
    69: "Does applying for the Research Fellowship in Sciences for Meritorious Students guarantee funding?",
    72: "Can the AICTE GATE/GPAT postgraduate scholarship continue beyond 24 months?",
    74: "How much does the Mukhya Mantri Sahara Yojana pay each month?",
    78: "Is the Gomant Vibhushan Award given annually or every two years?",
    95: "What Puducherry residence requirement applies to funeral assistance for Scheduled Caste families?",
    107: "What is the maximum insured amount under Niramaya health insurance?",
    108: "Does RMEWF medical assistance for ex-servicemen reimburse treatment at any private hospital?",
}


def display_text(text: str) -> str:
    for broken, fixed in {"â‚¹": "₹", "â€™": "’", "â€˜": "‘", "â€œ": "“",
                          "â€“": "–", "â€”": "—", "ï»¿": "", "â€‹": ""}.items():
        text = text.replace(broken, fixed)
    return text.strip()


def candidates(path: Path) -> list[dict]:
    output = []
    with pymupdf.open(path) as pdf:
        title = normalized_text(pdf[0].get_text().split("Are you sure")[0])
        if not 5 < len(title) < 180 or title in EXCLUDED_TITLES:
            return []
        for page_number, page in enumerate(pdf, 1):
            text = normalized_text(page.get_text())
            markers = list(re.finditer(r"Frequently Asked Questions", text))
            for marker in markers:
                if marker.start() < 700:
                    continue
                section = text[marker.end():].split("Sources And References")[0]
                questions = list(QUESTION.finditer(section))
                for index, match in enumerate(questions):
                    end = questions[index + 1].start() if index + 1 < len(questions) else len(section)
                    answer = section[match.end():end].strip()
                    question = match.group().strip()
                    if not 50 <= len(answer) <= 600 or "?" in answer or question.startswith("Can Be "):
                        continue
                    if not re.search(r"[.!]$", answer) or re.search(r"\.(?:Who|What|Under|Mention)|Who Is Eligible To$", answer):
                        continue
                    if re.search(r"\.\s+(?:Is|Are|Can|What|Who)\b", question):
                        continue
                    if any(marker in answer for marker in ["Something went wrong", "CancelSign", "©2024", "OkWas"]):
                        continue
                    # Include the source question in evidence; it resolves short yes/no answers.
                    evidence = section[match.start():end].strip()
                    if len(evidence) > 800:
                        continue
                    output.append({"question": display_text(f"Regarding {title}: {question}"),
                                   "reference_answer": display_text(answer), "category": "source_faq",
                                   "provenance": "Extracted source-authored FAQ; no LLM synthesis",
                                   "supporting_evidence": [{"filename": path.name, "page": page_number,
                                                            "quote": evidence}]})
                break
    return output


def build(data_dir: Path, count: int) -> list[dict]:
    paths = sorted(data_dir.rglob("*.pdf"))
    canonical = {}
    for path in paths:
        name = canonical_filename(path.name)
        if name not in canonical or path.name == name:
            canonical[name] = path
    # Retain existing cases, repair only the incorrect filename alias and encoding.
    anchors = ["financial assistance of 25", "40% subsidy on the purchases",
               "The support shall be towards", "The scheme will provide inpatient",
               "35% capital subsidy", "Out of the total premium", "loan subsidy of up to",
               "30,00,000", "The age limit for beneficiaries", "Children below 18"]
    cases = []
    for index, seed in enumerate(gold_standard):
        path = canonical[canonical_filename(seed["source_document"])]
        with pymupdf.open(path) as pdf:
            text = normalized_text(pdf[seed["page_number"] - 1].get_text())
        position = text.casefold().find(anchors[index].casefold())
        if position < 0:
            raise ValueError(f"Seed evidence anchor missing: {path.name} / {anchors[index]}")
        quote = text[max(0, position - (250 if index == 5 else 50)):position + 560]
        cases.append({"question": seed["question"], "reference_answer": seed["expected_answer"],
                      "category": "original_seed", "provenance": "Original manually authored case; PDF evidence added",
                      "supporting_evidence": [{"filename": path.name, "page": seed["page_number"], "quote": quote}]})
    remaining = [p for name, p in canonical.items()
                 if name not in {canonical_filename(c["source_document"]) for c in gold_standard}]
    random.Random(42).shuffle(remaining)
    for path in remaining:
        extracted = candidates(path)
        # At most two questions per additional document; broad source coverage.
        cases.extend(extracted[:2])
        if len(cases) >= count:
            break
    cases = cases[:count]
    if len(cases) != count:
        raise ValueError(f"Only {len(cases)} usable cases")
    for index, case in enumerate(cases, 1):
        if index in PARAPHRASES:
            case["original_source_question"] = case["question"]
            case["question"] = PARAPHRASES[index]
            case["category"] = "reviewed_paraphrase"
            case["provenance"] += "; question manually paraphrased before retrieval runs"
        if index == 6:
            case["original_reference_answer"] = case["reference_answer"]
            case["reference_answer"] += " The same PDF also says the premium is shared by central and state governments in a 50:50 ratio, so these statements are internally inconsistent."
            case["notes"] = "Known source contradiction; a grounded answer should flag it."
        evidence = case["supporting_evidence"][0]
        path = next(p for p in paths if p.name == evidence["filename"])
        evidence["pdf_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        case["id"] = f"scheme-{index:03d}"
        group = canonical_filename(evidence["filename"])
        case["split"] = "dev" if int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 5 == 0 else "test"
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--count", type=int, default=110)
    parser.add_argument("--output", type=Path, default=Path("evaluation/schemes_v1.jsonl"))
    args = parser.parse_args()
    cases = build(args.data_dir, args.count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cases), encoding="utf-8")
    print(f"Wrote {len(cases)} evidence-backed cases to {args.output}")


if __name__ == "__main__":
    main()
