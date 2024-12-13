import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/dataset_net.csv')
# Assuming 'protocol_distribution' is a Series object with protocol counts
protocol_distribution = df['protocol'].value_counts()

# Define a threshold (e.g., 5% of the total traffic)
threshold = 0.05

# Calculate total traffic
total_traffic = protocol_distribution.sum()

# Identify major protocols
major_protocols = protocol_distribution[protocol_distribution / total_traffic >= threshold]

# Group smaller protocols into "Others"
others = protocol_distribution[protocol_distribution / total_traffic < threshold].sum()

# Append the "Others" category to the major protocols
major_protocols['Others'] = others

# Plot the cleaned-up pie chart
plt.figure(figsize=(8, 8))
plt.pie(major_protocols, labels=major_protocols.index, autopct='%1.1f%%', startangle=140)
plt.title('Traffic Distribution by Protocol')
plt.show()
