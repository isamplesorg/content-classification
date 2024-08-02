import io

text_file = 'C:\Users\smrTu\OneDrive\Documents\Workspace\iSamples\training\trainingdata-part1.csv'

with open(text_file, 'r', errors='replace') as f:
    lines = f.readlines()

cleaned_lines = []
for line in lines:
    try:
        clean_line = line.encode('utf-8', errors='replace').decode('utf-8')
    except UnicodeDecodeError:
        clean_line = line.encode('utf-8', errors='replace').decode('utf-8')

    clean_line = clean_line.replace('\ufffd', '-')

    cleaned_lines.append(clean_line)

with io.open('clean_data.txt', 'w', encoding='utf-8') as f:
    f.writelines(cleaned_lines)