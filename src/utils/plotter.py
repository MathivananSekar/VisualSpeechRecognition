import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file with validation results
csv_file = "results/validation_results.csv"  # update the path if needed
df = pd.read_csv(csv_file)

# Compute average WER and CER per speaker
agg = df.groupby('speaker_id').agg({'wer': 'mean', 'cer': 'mean'}).reset_index()
print(agg)

# Prepare data for grouped bar chart
speakers = agg['speaker_id']
wer_values = agg['wer']
cer_values = agg['cer']

x = np.arange(len(speakers))  # label locations
width = 0.35  # width of each bar

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, wer_values, width, label='WER')
rects2 = ax.bar(x + width/2, cer_values, width, label='CER')

# Add labels, title, and custom x-axis tick labels, etc.
ax.set_ylabel('Error Rate')
ax.set_title('Average WER and CER per Speaker')
ax.set_xticks(x)
ax.set_xticklabels(speakers)
ax.legend()

# Add text for labels, if desired.
def autolabel(rects):
    """Attach a text label above each bar displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()
plt.savefig("average_error_rates.png", dpi=300)
plt.show()
