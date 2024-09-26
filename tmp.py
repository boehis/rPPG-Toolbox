import seaborn as sns
from statannotations.Annotator import Annotator
import matplotlib.pyplot as plt  # Import matplotlib for saving the plot

# Load dataset
df = sns.load_dataset("tips")
x = "day"
y = "total_bill"
order = ['Sun', 'Thur', 'Fri', 'Sat']

# Create a boxen plot instead of a box plot
ax = sns.boxenplot(data=df, x=x, y=y, order=order)

# Define pairs of categories you want to compare
pairs = [("Thur", "Fri"), ("Thur", "Sat"), ("Fri", "Sun")]

# Create the Annotator and configure it for the statistical test and annotation style
annotator = Annotator(ax, pairs, data=df, x=x, y=y, order=order)
annotator.configure(test='Mann-Whitney', text_format='star', loc='outside')

# Apply the statistical test and add the annotations
annotator.apply_and_annotate()

# Save the plot as an image file (e.g., PNG or PDF)
plt.savefig("boxen_plot_with_annotations.png", dpi=300, bbox_inches='tight')
# You can also save in other formats like PDF:
# plt.savefig("boxen_plot_with_annotations.pdf", bbox_inches='tight')

# Show the plot
plt.show()