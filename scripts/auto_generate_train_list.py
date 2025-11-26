import os
from pathlib import Path

DATA_DIR = Path("./data/UCF-Crime")

crime_labels = [
    "Normal", "Abuse", "Arrest", "Arson", "Assault", "Burglary",
    "Explosion", "Fighting", "RoadAccident", "Robbery",
    "Shooting", "Shoplifting", "Stealing", "Vandalism"
]

label_map = {name: idx for idx, name in enumerate(crime_labels)}

output = []

print("Scanning UCF-Crime folders...\n")

for crime in crime_labels:
    folder = DATA_DIR / crime
    if not folder.exists():
        print(f"[SKIP] Folder missing: {crime}")
        continue

    print(f"[OK] Scanning folder: {crime}")

    for file in os.listdir(folder):
        if file.lower().endswith(".mp4"):
            relative_path = f"{crime}/{file}"
            label_index = label_map[crime]
            output.append(f"{relative_path} {label_index}")
            print("  ➜ Found:", relative_path)

# Save output
with open("auto_train.txt", "w") as f:
    for line in output:
        f.write(line + "\n")

print("\n DONE!")
print("Generated file: auto_train.txt")
print(f"Total videos detected: {len(output)}")
