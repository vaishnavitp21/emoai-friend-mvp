from transformers import pipeline

def main():
    print("Loading local fallback sentiment model (no download needed)...")

    # This model is bundled with transformers and does not need internet
    emo_pipe = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

    samples = [
        "I am so happy and excited about my new job!",
        "I feel really sad and down today.",
        "That was hilarious — I can't stop laughing!",
        "I'm worried about my exam tomorrow.",
        "This is so relaxing and peaceful."
    ]

    print("\nClassifying sample sentences:\n")
    for s in samples:
        result = emo_pipe(s)[0]
        print(f"Input: {s}")
        print(f"  -> Label: {result['label']} (score: {result['score']:.4f})")
        print()

if __name__ == "__main__":
    main()
