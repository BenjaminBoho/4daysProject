import os

# Relative path from the script to the Dataset directory
relative_path = "../Dataset"

# Convert to absolute path
absolute_path = os.path.abspath(relative_path)

# Print the absolute path
print("Absolute path of the Dataset directory:", absolute_path)
