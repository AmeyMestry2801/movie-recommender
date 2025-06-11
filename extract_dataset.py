import zipfile
import os

zip_path = "ml-1m.zip"
extract_to = "ml-1m"

if not os.path.exists(extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        print("✅ Dataset extracted!")
else:
    print("📁 Already extracted.")
