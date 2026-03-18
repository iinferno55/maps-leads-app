import textwrap

from app import detect_owner_with_ollama


def make_business_cases() -> list[dict]:
    cases: list[dict] = []

    # 1–3: very easy solo cases, one clearly named person.
    cases.append(
        {
            "business_name": "Josh's Mobile Detailing",
            "reviews": [
                "Josh did an amazing job on my truck. Josh showed up on time and explained everything clearly.",
                "I highly recommend Josh for any mobile detailing. Josh was professional and honest.",
                "Called Josh last minute and he still squeezed me in. I'll be using Josh again.",
            ],
        }
    )
    cases.append(
        {
            "business_name": "Maria's Roofing",
            "reviews": [
                "Maria inspected my roof and gave a fair quote. Maria handled the whole project personally.",
                "When I called, Maria answered and scheduled me within two days. Maria was on site the entire time.",
                "If you need a small roofing job, call Maria. She clearly runs this herself.",
            ],
        }
    )
    cases.append(
        {
            "business_name": "Jeff's Auto Salon",
            "reviews": [
                "Jeff took care of my car inside and out. Jeff listened to my concerns and went above and beyond.",
                "I've been going to Jeff for years. Jeff always does the detailing himself.",
                "Jeff is a perfectionist. You can tell it's his own shop.",
            ],
        }
    )

    # 4–6: small team / spouse partner cases.
    cases.append(
        {
            "business_name": "Local Roots Landscaping",
            "reviews": [
                "Chris and Stacia Hays did a fantastic job on our yard. Chris walked the property and Stacia handled the design.",
                "We always work with Chris and Stacia at Local Roots. They are the owners and really care.",
                "Ask for Chris or Stacia when you call. They personally run the crew.",
            ],
        }
    )
    cases.append(
        {
            "business_name": "Jason & Teresa Cleaning",
            "reviews": [
                "Jason and Teresa showed up exactly on time and cleaned our place perfectly.",
                "We love working with Jason and Teresa. It's just the two of them and they do all the work.",
                "If you want a small family cleaning company, Jason and Teresa are it.",
            ],
        }
    )
    cases.append(
        {
            "business_name": "Jeff's Detail and Team",
            "reviews": [
                "Jeff and his team took care of years of dog hair in my SUV. Jeff explained every step.",
                "I always ask for Jeff when I call. Jeff and his team are the only ones I trust.",
                "Jeff and team detailed my truck after a long road trip. Jeff checked everything himself before I left.",
            ],
        }
    )

    # 7–10: harder multi-name / franchise style cases.
    cases.append(
        {
            "business_name": "Parker & Sons Plumbing",
            "reviews": [
                "Christian came out and fixed our leak quickly.",
                "Jason from Parker & Sons replaced our water heater.",
                "Angel did a great job snaking our main line.",
                "Mike came with another tech and they were both professional.",
            ],
        }
    )
    cases.append(
        {
            "business_name": "Citywide Roofing Pros",
            "reviews": [
                "Our sales rep Dave walked us through the options and then Nolan handled the install.",
                "Huge thanks to Sara in the office and the crew that came out.",
                "We worked with Marcus first and then a different foreman the day of the job.",
            ],
        }
    )
    cases.append(
        {
            "business_name": "Bright Smile Dental",
            "reviews": [
                "I saw Dr. Smith the first visit and Dr. Lee for the follow up.",
                "The hygienists Anna and Brooke were both wonderful.",
                "I've had appointments with three different dentists here and all were good.",
            ],
        }
    )
    cases.append(
        {
            "business_name": "Downtown HVAC Services",
            "reviews": [
                "The technician Greg arrived on time.",
                "Second visit we had someone named Luis.",
                "Most recently Amanda and her trainee fixed our AC.",
                "We've had a different person almost every time we call.",
            ],
        }
    )

    # Expand each reviews list into a single text block.
    for case in cases:
        case["reviews_text"] = "\n\n".join(case["reviews"])
    return cases


def main() -> None:
    cases = make_business_cases()
    print(f"Running owner detection on {len(cases)} synthetic businesses...\n")
    for idx, case in enumerate(cases, start=1):
        name = case["business_name"]
        reviews_text = case["reviews_text"]
        print("=" * 80)
        print(f"[{idx}] {name}")
        print("- Reviews sample -")
        print(textwrap.shorten(reviews_text.replace("\n", " "), width=260, placeholder=" ..."))
        det = detect_owner_with_ollama(reviews_text, name)
        print("\n- Detection -")
        print(f"owner_name : {det.get('owner_name')}")
        print(f"solo       : {det.get('solo')}")
        print(f"confidence : {det.get('confidence')}")
        print(f"reason     : {det.get('reason')}")
        print()


if __name__ == "__main__":
    main()

