import requests
from bs4 import BeautifulSoup
import json
import re

URL = "https://en.wikipedia.org/wiki/Artificial_intelligence"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/130.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

def main():
    resp = requests.get(URL, headers=HEADERS)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    title_tag = soup.find("h1")
    title = title_tag.get_text(strip=True) if title_tag else "NO TITLE FOUND"
    print("\nPAGE TITLE:", title)
    print("=" * 80)

    body = (
        soup.find("article")
        or soup.find("main")
        or soup.find("div", attrs={"role": "main"})
    )
    if not body:
        print("Could not find main article body.")
        return

    for tag in body.select("script, style, nav, footer, form"):
        tag.decompose()

    current_heading = "Introduction"
    content = {current_heading: []}

    for elem in body.descendants:
        name = getattr(elem, "name", None)

        if name in ("h2", "h3"):
            heading_text = elem.get_text(" ", strip=True)
            if heading_text:
                current_heading = heading_text
                content.setdefault(current_heading, [])

        elif name == "p":
            text = elem.get_text(" ", strip=True)

            # Remove reference markers like [4], [123], [a], [aa], [x]
            text = re.sub(r"\[[^\]]+\]", "", text)

            # Remove extra spaces
            text = re.sub(r"\s{2,}", " ", text).strip()

            if text:
                content.setdefault(current_heading, []).append(text)

    # 🔥 Remove headings that are completely empty
    content = {h: p for h, p in content.items() if p}

    # 🔥 Remove headings where ALL paragraphs are useless (end with ":" or too short)
    cleaned_content = {}
    for heading, paragraphs in content.items():
        meaningful_paras = []
        for p in paragraphs:
            # Skip useless lines like "Other textbooks:" / "Further reading:"
            if p.endswith(":") or len(p.split()) <= 4:
                continue
            meaningful_paras.append(p)

        if meaningful_paras:
            cleaned_content[heading] = meaningful_paras

    content = cleaned_content

    # Print console output
    for heading, paragraphs in content.items():
        print("\n", heading)
        print("\n".join(paragraphs))

    # Save to JSON
    with open("article.json", "w", encoding="utf-8") as f:
        json.dump(content, f, ensure_ascii=False, indent=4)

    print("\nSaved article.json successfully ✔")

if __name__ == "__main__":
    main()
