import argparse
import json
from pathlib import Path

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def _load_articles(path):
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict) and "articles" in data:
        return data["articles"]
    if isinstance(data, list):
        return data
    raise ValueError("Expected a list of articles or a dict with an 'articles' key.")


def _article_text(article):
    parts = []
    for key in ("title", "headline", "summary", "content", "text", "body"):
        value = article.get(key)
        if value:
            parts.append(str(value))
    return " ".join(parts).strip()


def run(input_path, output_path):
    analyzer = SentimentIntensityAnalyzer()
    articles = _load_articles(input_path)
    results = []
    for article in articles:
        text = _article_text(article)
        scores = analyzer.polarity_scores(text)
        results.append(
            {
                "text": text,
                "compound": scores["compound"],
                "positive": scores["pos"],
                "neutral": scores["neu"],
                "negative": scores["neg"],
                "meta": {k: v for k, v in article.items() if k not in {"content", "text", "body"}},
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)


def main():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run VADER sentiment on scraped articles.")
    parser.add_argument(
        "--input",
        default=str(script_dir / "articles.json"),
        help="Path to scraped articles JSON.",
    )
    parser.add_argument(
        "--output",
        default=str(script_dir / "sentiment.json"),
        help="Path to output sentiment JSON.",
    )
    args = parser.parse_args()
    run(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
