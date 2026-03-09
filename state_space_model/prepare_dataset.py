import numpy as np
from data_pipeline import TEPDataProcessor

SEQ_LEN = 200

processor = TEPDataProcessor(seq_len=SEQ_LEN)
X_seq, U_seq, y = processor.process("TEP_FaultFree_Training.RData")

print("Saving dataset using memory-mapped format...")

np.save("X_seq.npy", X_seq)
np.save("U_seq.npy", U_seq)
np.save("y.npy", y)

print("Saved:")
print("X_seq:", X_seq.shape)
print("U_seq:", U_seq.shape)
print("y:", y.shape)
