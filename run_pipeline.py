"""
End-to-end pipeline: feature engineering -> train & tune all models ->
evaluate the best one -> generate the Kaggle submission file.

Usage:
    python run_pipeline.py
"""
from src import train, evaluate, predict


def main():
    print("=" * 70)
    print("STEP 1/3: Training & tuning all models")
    print("=" * 70)
    train.main()

    print("\n" + "=" * 70)
    print("STEP 2/3: Evaluating the best model")
    print("=" * 70)
    evaluate.main()

    print("\n" + "=" * 70)
    print("STEP 3/3: Generating outputs/submission.csv")
    print("=" * 70)
    predict.main()

    print("\nPipeline complete. See reports/ and outputs/ for results.")


if __name__ == "__main__":
    main()
