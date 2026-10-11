from dotenv import load_dotenv

load_dotenv()
import torch
from transformers import AutoModel, AutoTokenizer

model_name = "ibm-research/MoLFormer-XL-both-10pct"
model = AutoModel.from_pretrained(
    model_name, deterministic_eval=True, trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model.eval()

smiles = ["Cn1c(=O)c2c(ncn2C)n(C)c1=O", "CC(=O)Oc1ccccc1C(=O)O"]
inputs = tokenizer(smiles, padding=True, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# Capture individual tensors step-by-step
print("--- Iterating Over Molecular Embeddings ---")
for idx, (smi, emb) in enumerate(zip(smiles, outputs.pooler_output)):
    # emb is an isolated 1D tensor representing a single chemical structure
    vct_list = emb.tolist()
    print(f"[{idx}] Captured embedding for: {smi}")
    print(f"    Embedding length: {len(emb)}")
    print(f"    Min value: {emb.min().item():.4f} | Max value: {emb.max().item():.4f}")