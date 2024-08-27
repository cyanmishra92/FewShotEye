import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def generate_heatmap(csv_file_path):
    # Load data from a CSV file
    data = pd.read_csv(csv_file_path, index_col=0)
    
    # Create a heatmap with a specified monochrome color map
    plt.figure(figsize=(10, 8))
    heatmap = sns.heatmap(data, annot=True, fmt=".1f", cmap="Blues", cbar_kws={'label': 'Performance'})
    
    # Add titles and labels
    plt.title('Monochrome Heatmap of Performance Data')
    plt.xlabel('n')
    plt.ylabel('k/n')
    
    # Save the plot to a PDF file
    plt.savefig("performance_heatmap.pdf", format='pdf')
    
    # Show the plot
    plt.show()

# Example usage
generate_heatmap('path_to_your_file.csv')

# Different color maps:
# 'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds', 'YlOrBr', 'YlOrRd'
# You can replace the 'Blues' in the cmap parameter with any of these to change the color theme.
