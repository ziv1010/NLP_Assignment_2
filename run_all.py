"""
Master runner — executes the full Hindi NLP pipeline.
Usage:
    python run_all.py              # Run all steps (subset mode by default)
    python run_all.py --full       # Run with full data
    python run_all.py --step 3     # Run only step 3
    python run_all.py --from_step 5  # Run from step 5 onwards
"""
import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_step(step_num, step_name, func):
    """Run a single pipeline step with timing."""
    print(f"\n{'#'*70}")
    print(f"# STEP {step_num}: {step_name}")
    print(f"{'#'*70}\n")
    start = time.time()
    try:
        func()
        elapsed = time.time() - start
        print(f"\n  ✓ Step {step_num} completed in {elapsed:.1f}s")
        return True
    except Exception as e:
        elapsed = time.time() - start
        print(f"\n  ✗ Step {step_num} FAILED after {elapsed:.1f}s: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Hindi NLP Pipeline Runner")
    parser.add_argument("--full", action="store_true",
                       help="Run with full data (default: subset mode)")
    parser.add_argument("--step", type=int, default=None,
                       help="Run only this step number")
    parser.add_argument("--from_step", type=int, default=1,
                       help="Start from this step number")
    parser.add_argument("--skip_download_pretrained", action="store_true",
                       help="Skip downloading pretrained vectors (step 4)")
    args = parser.parse_args()

    # Set mode
    import config
    if args.full:
        config.SUBSET_MODE = False
        print("[run_all] FULL DATA MODE")
    else:
        config.SUBSET_MODE = True
        print("[run_all] SUBSET MODE (use --full for full data)")

    # Define steps
    steps = [
        (1, "Download Hindi Corpus", lambda: __import__('step1_download_corpus').download_corpus()),
        (2, "Preprocess Corpus", lambda: __import__('step2_preprocess_corpus').preprocess_corpus()),
        (3, "Train Word Vectors (FastText + Word2Vec)", lambda: __import__('step3_train_word_vectors').train_all()),
        (4, "Download Pretrained Vectors", lambda: __import__('step4_download_pretrained').download_pretrained()),
        (5, "Compare Word Vectors", lambda: __import__('step5_compare_vectors').compare_vectors()),
        (6, "Prepare Classification Data", lambda: __import__('step6_prepare_classification_data').prepare_data()),
        (7, "Train LSTM Classifiers", lambda: __import__('step8_train_lstm').main()),
        (8, "Evaluate LSTM Classifiers", lambda: __import__('step9_evaluate_lstm').main()),
    ]

    # Filter steps
    if args.step is not None:
        steps = [(n, name, f) for n, name, f in steps if n == args.step]
    else:
        steps = [(n, name, f) for n, name, f in steps if n >= args.from_step]

    if args.skip_download_pretrained:
        steps = [(n, name, f) for n, name, f in steps if n != 4]

    print(f"\n[run_all] Will execute {len(steps)} steps")
    total_start = time.time()
    results = []

    for step_num, step_name, func in steps:
        success = run_step(step_num, step_name, func)
        results.append((step_num, step_name, success))
        if not success:
            print(f"\n  Pipeline stopped at step {step_num}. Fix the error and re-run with --from_step {step_num}")
            break

    total_elapsed = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"PIPELINE SUMMARY (Total time: {total_elapsed:.1f}s)")
    print(f"{'='*70}")
    for step_num, step_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"  Step {step_num}: {step_name:<45} {status}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
