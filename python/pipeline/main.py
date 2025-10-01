import argparse
from voice_bio_pipeline import VoiceBiometricsPipeline


def main():
    parser = argparse.ArgumentParser(description="Speaker verification using VoiceBiometricsPipeline")
    parser.add_argument("wav1", type=str, help="Enrollment audio file (wav1)")
    parser.add_argument("wav2", type=str, help="Test audio file (wav2)")
    args = parser.parse_args()

    pipeline = VoiceBiometricsPipeline()

    print("Enrolling speaker...")
    enrollment_embedding, chunks = pipeline.extract_embedding(args.wav1)

    if enrollment_embedding is None:
        print("No speech detected in the enrollment audio")
        return

    print("\nVerifying speaker...")
    test_embedding, _ = pipeline.extract_embedding(args.wav2)

    if test_embedding is None:
        print("No speech detected in the test audio")
        return

    similarity = pipeline.compute_similarity(enrollment_embedding, test_embedding)
    print(f"\nSimilarity score: {similarity:.4f}")

    if similarity > 0.5:
        print("The speakers are the same")
    else:
        print("The speakers are different")


if __name__ == "__main__":
    main()