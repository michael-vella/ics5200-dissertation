from dotenv import load_dotenv

load_dotenv()
import torch
from transformers import AutoModel, AutoTokenizer

model_name = "DeepChem/ChemBERTa-77M-MLM"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.eval()

smiles = ["Cn1c(=O)c2c(ncn2C)n(C)c1=O", "CC(=O)Oc1ccccc1C(=O)O"]
inputs = tokenizer(smiles, padding=True, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# ChemBERTa is an MLM-pretrained RoBERTa, so its pooler head is untrained;
# mean-pool the token embeddings (ignoring padding) to get one vector per molecule.
mask = inputs["attention_mask"].unsqueeze(-1).float()
embeddings = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)

print("--- Iterating Over Molecular Embeddings ---")
for idx, (smi, emb) in enumerate(zip(smiles, embeddings)):
    print(f"[{idx}] Captured embedding for: {smi}")
    print(f"    Embedding length: {len(emb)}")
    print(f"    Min value: {emb.min().item():.4f} | Max value: {emb.max().item():.4f}")
