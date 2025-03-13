import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_distance_distribution(csv_file, i_value, j_value):
    # Load the CSV file
    df = pd.read_csv(csv_file, header=None, names=["i", "j", "edge1", "edge2", "distance"])

    # Filter data for the given i and j
    filtered_df = df[(df["i"] == i_value) & (df["j"] == j_value)]

    if filtered_df.empty:
        print(f"No data found for i={i_value}, j={j_value}")
        return

    # Plot the distance distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(filtered_df["distance"], bins=70, kde=True, color='firebrick')
    plt.xlabel("Distance")
    plt.ylabel("Frequency")
    plt.title(f"Distance Distribution for i={i_value}, j={j_value}")
    sns.despine()
    plt.grid(False)
    plt.show()

if __name__ == "__main__":
    # Use WebAgg backend
    plt.switch_backend("WebAgg")

    plot_distance_distribution("Glove_sampl.csv", 24, 0)