import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import csv

from src.inference.predict import predict_lip_reading
from src.utils.validation_utils import compute_wer, compute_cer, parse_align_file

def validate_all_speakers(data_dir, checkpoint_path, results_dir):
    """
    data_dir: "data/raw"
    For each speaker s1..s10, compute average WER/CER.
    Write a CSV row for each video file prediction.
    Also store plots in results_dir.
    """
    os.makedirs(results_dir, exist_ok=True)
    
    csv_path = os.path.join(results_dir, "validation_results.csv")
    header = ["speaker_id", "video", "ref_text", "hyp_text", "wer", "cer"]
    # Open CSV in write mode (overwrite any previous file)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
    
    speakers = [f"s{i}" for i in range(1, 11)]
    speaker_wer = []
    speaker_cer = []
    
    # We'll also keep per-speaker metrics in lists.
    for spk in speakers:
        print(f"Validating speaker: {spk}")
        video_dir = os.path.join(data_dir, spk, "videos")
        align_dir = os.path.join(data_dir, spk, "alignments")
    
        mpg_files = glob.glob(os.path.join(video_dir, "*.mpg"))
        if not mpg_files:
            print(f"No .mpg files for {spk}, skipping.")
            speaker_wer.append(0)
            speaker_cer.append(0)
            continue
        
        wer_list = []
        cer_list = []
    
        for video_path in mpg_files:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            align_path = os.path.join(align_dir, f"{video_name}.align")
            if not os.path.exists(align_path):
                print(f"Missing align file for {video_name}, skipping.")
                continue
            
            # Parse reference text from alignment file
            ref_text = parse_align_file(align_path)
    
            # Predict using the model
            hyp_text = predict_lip_reading(video_path, checkpoint_path)
    
            # Compute WER and CER
            wer_val = compute_wer(ref_text, hyp_text)
            cer_val = compute_cer(ref_text, hyp_text)
    
            wer_list.append(wer_val)
            cer_list.append(cer_val)
    
            # Write result for this video to CSV (append mode)
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
    
        # Compute speaker-level averages
        avg_wer = np.mean(wer_list) if wer_list else 0
        avg_cer = np.mean(cer_list) if cer_list else 0
        speaker_wer.append(avg_wer)
        speaker_cer.append(avg_cer)
    
    print(f"Validation results CSV saved to: {csv_path}")
    
    # Now plot average WER/CER for each speaker
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
