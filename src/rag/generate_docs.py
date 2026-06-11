import json
import random
from pathlib import Path

random.seed(42)

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

AUTO_POLICY = """POLICY DOCUMENT - AUTO INSURANCE
Policy Number: {policy_no}
Policyholder: {holder}
Effective Date: 2024-01-15
Expiry Date: 2025-01-14

SECTION 1 - COVERAGE
This policy provides comprehensive auto insurance coverage including
third-party liability, collision, theft, and fire damage.

SECTION 2 - PREMIUMS AND DEDUCTIBLES
Annual Premium: ${premium}
Collision Deductible: ${collision_ded}
Comprehensive Deductible: ${comp_ded}
A deductible is the amount you pay before insurance covers the rest.

SECTION 3 - CLAIMS PROCEDURE
To file a claim:
1. Report the incident within 72 hours by calling 1-800-CLAIMS.
2. Provide your policy number and description of incident.
3. Submit a police report for theft or injury incidents.
Claims are processed within 14 business days.

SECTION 4 - EXCLUSIONS
This policy does NOT cover:
- Driving under influence of alcohol or drugs.
- Commercial use of vehicle.
- Wear and tear or mechanical breakdown.
"""

HOME_POLICY = """POLICY DOCUMENT - HOME INSURANCE
Policy Number: {policy_no}
Policyholder: {holder}
Property: {address}
Effective Date: 2024-03-01

SECTION 1 - DWELLING COVERAGE
Covers direct physical loss up to ${dwelling_limit}.
Covered perils: fire, lightning, windstorm, hail, vandalism.
Flood and earthquake require separate policies.

SECTION 2 - PERSONAL PROPERTY
Covered up to ${personal_limit}.
Deductible: ${deductible}

SECTION 3 - LIABILITY
Personal liability coverage of ${liability_limit}.
Covers bodily injury or property damage caused to others.

SECTION 4 - CLAIMS
Claims must be reported within 30 days of discovery.
"""

FAQS = [
    ("How long does it take to process an auto claim?",
     "Auto insurance claims are processed within 14 business days after all documents are received."),

    ("What is a deductible?",
     "A deductible is the amount you pay out of pocket before insurance covers the rest. For example with a $500 deductible on a $3000 repair, you pay $500 and insurer pays $2500."),

    ("Is flood damage covered under homeowners policy?",
     "No. Standard homeowners insurance excludes flood damage. You need a separate flood insurance policy."),

    ("Can I cancel my policy at any time?",
     "Yes. Most policies allow cancellation with 30 days written notice. Unused premium is refunded minus a small fee."),

    ("What happens if I miss a premium payment?",
     "Most policies provide a 30 day grace period. If payment is not received the policy will lapse and coverage ends."),
]

NAMES = ["Alex Patel", "Jordan Garcia", "Sam Smith", "Taylor Nguyen", "Morgan Kim"]
ADDRESSES = ["123 Maple St", "456 Oak Ave", "789 Pine Rd", "321 Elm Blvd"]


def rand_policy_no(prefix):
    return f"{prefix}-{random.randint(100000, 999999)}"


def gen_auto(i):
    return AUTO_POLICY.format(
        policy_no=rand_policy_no("AUTO"),
        holder=random.choice(NAMES),
        premium=random.choice([850, 1200, 1450, 1800]),
        collision_ded=random.choice([250, 500, 1000]),
        comp_ded=random.choice([100, 250, 500]),
    )


def gen_home(i):
    return HOME_POLICY.format(
        policy_no=rand_policy_no("HOME"),
        holder=random.choice(NAMES),
        address=random.choice(ADDRESSES),
        dwelling_limit=random.choice([250000, 400000, 600000]),
        personal_limit=random.choice([50000, 100000]),
        deductible=random.choice([500, 1000, 2500]),
        liability_limit=random.choice([100000, 300000]),
    )


def main():
    generated = []

    for i in range(3):
        for kind, fn in [("auto", gen_auto), ("home", gen_home)]:
            content = fn(i)
            filename = f"{kind}_policy_{i+1}.txt"
            path = OUTPUT_DIR / filename
            path.write_text(content)
            generated.append({"file": filename, "type": kind})

    faq_content = "FREQUENTLY ASKED QUESTIONS\n\n"
    for q, a in FAQS:
        faq_content += f"Q: {q}\nA: {a}\n\n"

    (OUTPUT_DIR / "faqs.txt").write_text(faq_content)
    generated.append({"file": "faqs.txt", "type": "faq"})

    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(generated, indent=2))

    print(f"Generated {len(generated)} documents in {OUTPUT_DIR}")
    for g in generated:
        print(f"  - {g['file']}")


if __name__ == "__main__":
    main()