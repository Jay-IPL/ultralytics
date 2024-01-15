import csv

# Path to your existing CSV file
input_file_path = 'your_input_file.csv'

# Path to the new file to be created (can be the same as input for overwriting)
output_file_path = 'your_output_file.csv'

# Read the data, skipping empty rows
with open(input_file_path, 'r', newline='', encoding='utf-8') as input_file:
    reader = csv.reader(input_file)
    data = [row for row in reader if any(row)]

# Write the data back to a new file
with open(output_file_path, 'w', newline='', encoding='utf-8') as output_file:
    writer = csv.writer(output_file)
    writer.writerows(data)
