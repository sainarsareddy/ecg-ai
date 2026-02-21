import sys

import numpy as np
import wfdb

sys.path.append("..")
from phase2_features.extractor import extract_features

record = wfdb.rdrecord('100', pn_dir='mitdb')
signal = record.p_signal[:, 0]
segment = signal[:1800].tolist()

print("Signal sample:", segment[:5])
print("\nExtracted Features:")
features = extract_features(segment)
for key, value in features.items():
    print(f"  {key}: {value}")
print("\nSignal for Swagger UI:")
print(segment)