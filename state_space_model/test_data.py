from data_pipeline import TEPDataProcessor

processor = TEPDataProcessor(seq_len=200)

X_seq, U_seq, y = processor.process(
    "TEP_FaultFree_Training.RData"
)

print("State sequence shape:", X_seq.shape)
print("Control sequence shape:", U_seq.shape)
print("Target shape:", y.shape)