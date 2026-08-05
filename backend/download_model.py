from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os

# Load from Hugging Face
model = AutoModelForSequenceClassification.from_pretrained("helalsalloum/arabic-fact-verifier")
tokenizer = AutoTokenizer.from_pretrained("helalsalloum/arabic-fact-verifier")

# Save to the expected location
output_dir = "../models/verifier"
os.makedirs(output_dir, exist_ok=True)

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"✅ Model saved to: {output_dir}")
print(f"📁 Files: {os.listdir(output_dir)}")