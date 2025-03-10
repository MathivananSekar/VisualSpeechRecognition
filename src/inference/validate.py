import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import csv

from src.inference.predict import predict_lip_reading
from src.utils.validation_utils import compute_wer, compute_cer, parse_align_file

def validate_all_speakers(data_dir, checkpoint_path, results_dir):
    """
    For each speaker s1..s10 in data_dir, compute WER/CER for each video,
    and write results (ref_text, hyp_text) to a CSV. Also plot average WER/CER.
    """
    os.makedirs(results_dir, exist_ok=True)

    csv_path = os.path.join(results_dir, "validation_results.csv")
    header = ["speaker_id", "video", "ref_text", "hyp_text", "wer", "cer"]

    # Overwrite or create the CSV file, writing headers first
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

    # Example: s1..s10
    speakers = ["s34"]
    speaker_wer = []
    speaker_cer = []

    for spk in speakers:
        print(f"Validating speaker: {spk}")
        video_dir = os.path.join(data_dir, spk, "videos")
        align_dir = os.path.join(data_dir, spk, "alignments")

        mpg_files = glob.glob(os.path.join(video_dir, "*.mpg"))
        if not mpg_files:
            print(f"No .mpg files for {spk}, skipping.")
            speaker_wer.append(0.0)
            speaker_cer.append(0.0)
            continue

        wer_list = []
        cer_list = []

        for video_path in mpg_files:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            align_path = os.path.join(align_dir, f"{video_name}.align")
            if not os.path.exists(align_path):
                print(f"Missing align file for {video_name}, skipping.")
                continue

            # 1) Parse reference text
            ref_text = parse_align_file(align_path)

            # 2) Get hypothesis from predict_lip_reading
            hyp_text = predict_lip_reading(video_path, checkpoint_path)

            # If hyp_text is a tuple, we only want the final prediction as a string
            # e.g. (greedy_pred, fused_pred). We assume the second element is final:
            if isinstance(hyp_text, tuple):
                # If second element is a list of words, join them
                if len(hyp_text) > 1:
                    if isinstance(hyp_text[1], list):
                        hyp_text = " ".join(hyp_text[1])
                    else:
                        hyp_text = str(hyp_text[1])
                else:
                    # fallback if only one element
                    if isinstance(hyp_text[0], list):
                        hyp_text = " ".join(hyp_text[0])
                    else:
                        hyp_text = str(hyp_text[0])
            elif isinstance(hyp_text, list):
                # If we got a list of words
                hyp_text = " ".join(hyp_text)
            else:
                # It's presumably already a string
                hyp_text = str(hyp_text)

            # 3) Compute WER/CER
            wer_val = compute_wer(ref_text, hyp_text)
            cer_val = compute_cer(ref_text, hyp_text)
            wer_list.append(wer_val)
            cer_list.append(cer_val)

            # 4) Write CSV row
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    spk,
                    video_name,
                    ref_text,
                    hyp_text,
                    f"{wer_val:.4f}",
                    f"{cer_val:.4f}"
                ])

            print(f"Processed {video_name}: WER={wer_val:.4f}, CER={cer_val:.4f}")

        # Speaker-level averages
        avg_wer = np.mean(wer_list) if wer_list else 0.0
        avg_cer = np.mean(cer_list) if cer_list else 0.0
        speaker_wer.append(avg_wer)
        speaker_cer.append(avg_cer)

    print(f"Validation results CSV saved to: {csv_path}")

    # Plot average WER/CER across speakers
    x_ticks = range(len(speakers))

    # WER Plot
    plt.figure()
    plt.bar(x_ticks, speaker_wer)
    plt.xticks(x_ticks, speakers)
    plt.title("Average WER per Speaker")
    plt.ylabel("WER")
    plt.xlabel("Speaker")
    wer_plot_path = os.path.join(results_dir, "wer_plot.png")
    plt.savefig(wer_plot_path)
    plt.close()
    print(f"WER plot saved to {wer_plot_path}")

    # CER Plot
    plt.figure()
    plt.bar(x_ticks, speaker_cer)
    plt.xticks(x_ticks, speakers)
    plt.title("Average CER per Speaker")
    plt.ylabel("CER")
    plt.xlabel("Speaker")
    cer_plot_path = os.path.join(results_dir, "cer_plot.png")
    plt.savefig(cer_plot_path)
    plt.close()
    print(f"CER plot saved to {cer_plot_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Validate on all speakers and compute WER/CER.")
    parser.add_argument("--data_dir", type=str, default="data/raw", help="Path to raw GRID data.")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/epoch_50.pth",
                        help="Path to model checkpoint.")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to store plots and CSV.")
    args = parser.parse_args()

    validate_all_speakers(args.data_dir, args.checkpoint_path, args.results_dir)
