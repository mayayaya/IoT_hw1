import pandas as pd

# Load the dataset to inspect its contents
file_path = r"C:\Users\User\Desktop\Maya\IoT\Hw_01\Hw_1-2\BodyFatDataset.csv"
data = pd.read_csv(file_path)

# Display the first few rows to understand the structure
print(data.head(10))