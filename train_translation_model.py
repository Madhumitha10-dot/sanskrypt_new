"""
SansKrypt Translation Model Fine-Tuning & Training Pipeline
Trains and fine-tunes a Sanskrit -> English translation model using parallel Sanskrit corpora.
"""
import os
import sys
import logging
from typing import List, Dict, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AdamW,
    get_linear_schedule_with_warmup
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("TrainTranslator")

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_MODEL_DIR = os.path.join(BASE_DIR, "models", "sanskrypt_translator")
os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)

# Curated High-Quality Parallel Sanskrit-English Classical Corpus
PARALLEL_TRAINING_CORPUS: List[Tuple[str, str]] = [
    # Philosophical & Upanishadic
    ("सत्यमेव जयते नानृतम्", "Truth alone triumphs, not falsehood."),
    ("अहं ब्रह्मास्मि", "I am the Supreme Reality (Brahman)."),
    ("तत्त्वमसि", "That Thou Art."),
    ("कर्मण्येवाधिकारस्ते मा फलेषु कदाचन", "You have a right to perform your duty, but never to the fruits of action."),
    ("योगः कर्मसु कौशलम्", "Yoga is excellence and dexterity in action."),
    ("वसुधैव कुटुम्बकम्", "The whole world is one single family."),
    ("सर्वे भवन्तु सुखिनः सर्वे सन्तु निरामयाः", "May all beings be happy, may all beings be free from illness."),
    ("असतो मा सद्गमय तमसो मा ज्योतिर्गमय", "Lead me from the unreal to the real, lead me from darkness to light."),
    ("विद्या ददाति विनयं विनयाद् याति पात्रताम्", "Knowledge bestows humility; from humility one attains worthiness."),
    ("उद्धरेदात्मनात्मानं नात्मानमवसादयेत्", "One must elevate oneself by one's own mind, and not degrade oneself."),
    
    # Ayurvedic & Medical
    ("वायुः पित्तं कफश्चेति त्रयो दोषाः समासतः", "Vata, Pitta, and Kapha are the three bodily humors in brief."),
    ("समदोषः समाग्निश्च समधातुमलक्रियः प्रसन्नत्मेन्द्रियमनाः स्वस्थ इत्यभिधीयते", "One who has balanced doshas, balanced digestive fire, balanced tissues and excretions, and a tranquil soul, senses, and mind is called healthy."),
    ("शरीरमाद्यं खलु धर्मसाधनम्", "The physical body is indeed the prime instrument for fulfilling righteousness."),
    ("आहारशुद्धौ सत्त्वशुद्धिः", "When food is pure, the mind and inner essence become pure."),
    ("हिताहारविहारसेवी समीक्ष्यकारी विषयेष्वसक्तः", "One who takes wholesome diet and lifestyle, acts with reflection, and remains unattached to sensory indulgence remains healthy."),
    
    # Astronomical & Mathematical
    ("शून्यं सर्वत्र सम्पूर्णा", "Zero is complete everywhere."),
    ("मखि भखि फखि धखि णखि", "The alphabetical computation table of sine differences in astronomical trigonometry."),
    ("भूग्रहभ्रमणं वदन्ति", "They describe the orbital revolution of the earth and celestial planets."),
    ("चक्रवत् परिवर्तन्ते सुखानि च दुःखानि च", "Happiness and sorrow revolve continuously like the spokes of a wheel."),
    ("अङ्कानां वामतो गतिः", "The progression of digits in calculation proceeds from right to left."),
    
    # Literature & Epics
    ("अहिंसा परमो धर्मः", "Non-violence is the highest ethical duty."),
    ("जननी जन्मभूमिश्च स्वर्गादपि गरीयसी", "Mother and motherland are far greater even than heaven."),
    ("यत्र नार्यस्तु पूज्यन्ते रमन्ते तत्र देवताः", "Where women are honored, there the gods rejoice."),
    ("धर्मो रक्षति रक्षितः", "Dharma protects those who uphold and protect it."),
    ("काव्यशास्त्रविनोदेन कालो गच्छति धीमताम्", "The time of the wise passes in the enjoyment of poetry and sciences.")
]


class SanskritTranslationDataset(Dataset):
    """PyTorch Dataset for parallel Sanskrit -> English pairs."""
    def __init__(self, pairs: List[Tuple[str, str]], tokenizer, max_length: int = 128):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        src_text = f"translate Sanskrit to English: {src.strip()}"
        tgt_text = tgt.strip()

        src_encoding = self.tokenizer(
            src_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        tgt_encoding = self.tokenizer(
            tgt_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        labels = tgt_encoding["input_ids"].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100  # Ignore padding in loss

        return {
            "input_ids": src_encoding["input_ids"].squeeze(0),
            "attention_mask": src_encoding["attention_mask"].squeeze(0),
            "labels": labels
        }


def load_additional_huggingface_dataset() -> List[Tuple[str, str]]:
    """Attempts to load additional parallel pairs from Hugging Face or local cache."""
    extra_pairs = []
    try:
        from datasets import load_dataset
        logger.info("Checking for 'snskrt/Sanskrit_OCR_Parallel_Corpus' or parallel translation datasets...")
        # Load sample if accessible
        ds = load_dataset("snskrt/Sanskrit_OCR_Parallel_Corpus", split="train[:50]", trust_remote_code=True)
        for item in ds:
            sanskrit = item.get("sanskrit") or item.get("text") or item.get("source")
            english = item.get("english") or item.get("translation") or item.get("target")
            if sanskrit and english and len(sanskrit.strip()) > 3 and len(english.strip()) > 3:
                extra_pairs.append((sanskrit.strip(), english.strip()))
        logger.info(f"Loaded {len(extra_pairs)} additional pairs from dataset.")
    except Exception as e:
        logger.info(f"Note: Using rich embedded classical parallel corpus ({len(PARALLEL_TRAINING_CORPUS)} pairs). External info: {str(e)}")
    return extra_pairs


def train_model(
    base_model_name: str = "google/flan-t5-small",
    epochs: int = 5,
    batch_size: int = 4,
    learning_rate: float = 3e-4
):
    """
    Fine-tunes the Seq2Seq translation model on Sanskrit -> English parallel pairs.
    """
    logger.info(f"Loading base model and tokenizer: {base_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name, torch_dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Training on device: {device}")

    # Combine corpus
    corpus = list(PARALLEL_TRAINING_CORPUS)
    extra_pairs = load_additional_huggingface_dataset()
    corpus.extend(extra_pairs)

    logger.info(f"Total training parallel pairs: {len(corpus)}")
    dataset = SanskritTranslationDataset(corpus, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )

    model.train()
    logger.info(f"Starting fine-tuning for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for batch in dataloader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(len(dataloader), 1)
        logger.info(f"Epoch {epoch}/{epochs} - Average Loss: {avg_loss:.4f}")

    # Save fine-tuned checkpoint
    logger.info(f"Saving fine-tuned Sanskrit translator to: {OUTPUT_MODEL_DIR}")
    model.save_pretrained(OUTPUT_MODEL_DIR)
    tokenizer.save_pretrained(OUTPUT_MODEL_DIR)
    logger.info("Translation model training & export completed successfully!")


if __name__ == "__main__":
    train_model(epochs=4)
