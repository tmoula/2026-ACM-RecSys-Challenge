"""
Fine-tune a dense retriever with contrastive learning on TalkPlay train conversations.

The training pairs match inference:
  query  = conversation history (music turns expanded to track metadata)
  passage = track metadata fields used by DenseRetriever at index time

Example (Colab):
  python train_dense_retriever.py --epochs 2 --batch-size 16
"""

import argparse
import json
import os
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

CORPUS_TYPES = ["track_name", "artist_name", "album_name", "tag_list", "release_date"]


def track_to_passage(track: Dict) -> str:
    lines = []
    for field in CORPUS_TYPES:
        value = track.get(field, "")
        if isinstance(value, list):
            value = ", ".join(str(item) for item in value)
        lines.append(f"{field}: {value}")
    return "\n".join(lines)


def expand_turn(role: str, content: str, tracks: Dict[str, Dict]) -> str:
    if role == "music":
        track = tracks.get(content)
        if track is not None:
            return f"assistant: {track_to_passage(track)}"
    return f"{role}: {content}"


def user_profile_to_text(user_profile: Dict | None) -> str:
    """Format user demographic fields as retrieval context."""
    if not user_profile:
        return ""
    fields = [
        ("age_group", user_profile.get("age_group", "")),
        ("gender", user_profile.get("gender", "")),
        ("country_name", user_profile.get("country_name", "")),
    ]
    lines = [f"{key}: {value}" for key, value in fields if str(value).strip()]
    if not lines:
        return ""
    return "user_profile:\n" + "\n".join(lines)


def infer_current_conditions(history: List[str]) -> str:
    """
    Infer lightweight condition hints from the conversation text.
    This only uses mentions present in the dialogue (no external APIs).
    """
    text = "\n".join(history).lower()
    tags: List[str] = []

    if any(word in text for word in ["morning", "sunrise", "early"]):
        tags.append("time_of_day: morning")
    if any(word in text for word in ["afternoon", "noon"]):
        tags.append("time_of_day: afternoon")
    if any(word in text for word in ["evening", "night", "midnight", "late"]):
        tags.append("time_of_day: night")

    if any(word in text for word in ["rain", "rainy", "storm", "thunder"]):
        tags.append("weather: rainy_or_stormy")
    if any(word in text for word in ["sunny", "clear sky", "bright day"]):
        tags.append("weather: sunny")
    if any(word in text for word in ["snow", "snowy", "blizzard"]):
        tags.append("weather: snowy")

    if any(word in text for word in ["home", "house", "bedroom"]):
        tags.append("location_hint: home")
    if any(word in text for word in ["car", "driving", "road trip", "commute", "train", "bus"]):
        tags.append("location_hint: transit")
    if any(word in text for word in ["gym", "workout", "running"]):
        tags.append("location_hint: workout")
    if any(word in text for word in ["office", "work", "study", "library"]):
        tags.append("location_hint: work_or_study")

    if not tags:
        return ""
    return "current_conditions:\n" + "\n".join(tags)


class ConversationTrackDataset(Dataset):
    """(conversation context, target track passage) pairs from conversation music turns."""

    def __init__(
        self,
        split: str = "train",
        max_samples: int | None = None,
        label: str | None = None,
    ) -> None:
        conversations = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split=split)
        users = {
            row["user_id"]: row
            for row in load_dataset("talkpl-ai/TalkPlayData-Challenge-User-Metadata", split="all_users")
        }
        tracks = {
            row["track_id"]: row
            for row in load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata", split="all_tracks")
        }
        self.examples: List[Tuple[str, str]] = []

        for session in conversations:
            history: List[str] = []
            profile_text = user_profile_to_text(users.get(session.get("user_id")))
            for turn in session["conversations"]:
                role = turn["role"]
                content = turn["content"]
                if role == "music":
                    track = tracks.get(content)
                    if track is not None:
                        conditions_text = infer_current_conditions(history)
                        query_parts = [part for part in [profile_text, conditions_text, "\n".join(history)] if part]
                        query = "\n\n".join(query_parts)
                        passage = track_to_passage(track)
                        if query.strip() and passage.strip():
                            self.examples.append((query, passage))
                history.append(expand_turn(role, content, tracks))

        if max_samples is not None:
            self.examples = self.examples[:max_samples]

        split_label = label or split
        print(f"{split_label} pairs: {len(self.examples)}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Tuple[str, str]:
        return self.examples[index]


def collate(batch, tokenizer, query_prefix: str, passage_prefix: str, max_length: int):
    queries, passages = zip(*batch)
    query_batch = tokenizer(
        [query_prefix + query for query in queries],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    passage_batch = tokenizer(
        [passage_prefix + passage for passage in passages],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return query_batch, passage_batch


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def contrastive_loss(
    query_emb: torch.Tensor,
    passage_emb: torch.Tensor,
    temperature: float = 0.05,
) -> torch.Tensor:
    query_emb = F.normalize(query_emb, dim=1)
    passage_emb = F.normalize(passage_emb, dim=1)
    logits = torch.matmul(query_emb, passage_emb.T) / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, labels)


@torch.no_grad()
def evaluate_loader(
    model: AutoModel,
    loader: DataLoader,
    device: str,
    temperature: float,
) -> float:
    """Average contrastive loss over a dataloader (validation)."""
    if len(loader) == 0:
        return float("nan")

    model.eval()
    total_loss = 0.0
    for query_batch, passage_batch in loader:
        query_batch = {key: value.to(device) for key, value in query_batch.items()}
        passage_batch = {key: value.to(device) for key, value in passage_batch.items()}

        query_outputs = model(**query_batch)
        passage_outputs = model(**passage_batch)
        query_emb = mean_pool(query_outputs.last_hidden_state, query_batch["attention_mask"])
        passage_emb = mean_pool(passage_outputs.last_hidden_state, passage_batch["attention_mask"])
        total_loss += contrastive_loss(query_emb, passage_emb, temperature=temperature).item()

    model.train()
    return total_loss / len(loader)


def train(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_dataset = ConversationTrackDataset(
        split=args.train_split,
        max_samples=args.max_samples,
        label="train",
    )
    if len(train_dataset) == 0:
        raise RuntimeError("No training examples found.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModel.from_pretrained(args.model_name).to(device)
    model.train()

    val_loader = None
    if args.val_split:
        val_dataset = ConversationTrackDataset(
            split=args.val_split,
            max_samples=args.max_val_samples,
            label=f"val ({args.val_split})",
        )
        if len(val_dataset) > 0:
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=lambda batch: collate(
                    batch,
                    tokenizer,
                    query_prefix=args.query_prefix,
                    passage_prefix=args.passage_prefix,
                    max_length=args.max_length,
                ),
            )
        else:
            print(f"Warning: val split {args.val_split!r} is empty; skipping validation.")

    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate(
            batch,
            tokenizer,
            query_prefix=args.query_prefix,
            passage_prefix=args.passage_prefix,
            max_length=args.max_length,
        ),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(100, total_steps // 10),
        num_training_steps=total_steps,
    )

    epoch_metrics: List[Dict[str, float | int]] = []
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for step, (query_batch, passage_batch) in enumerate(loader, start=1):
            query_batch = {key: value.to(device) for key, value in query_batch.items()}
            passage_batch = {key: value.to(device) for key, value in passage_batch.items()}

            query_outputs = model(**query_batch)
            passage_outputs = model(**passage_batch)
            query_emb = mean_pool(query_outputs.last_hidden_state, query_batch["attention_mask"])
            passage_emb = mean_pool(passage_outputs.last_hidden_state, passage_batch["attention_mask"])
            loss = contrastive_loss(query_emb, passage_emb, temperature=args.temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

            if step % 20 == 0 or step == len(loader):
                print(f"epoch {epoch + 1}/{args.epochs} step {step}/{len(loader)} loss={loss.item():.4f}")

        train_avg = epoch_loss / max(len(loader), 1)
        val_avg = (
            evaluate_loader(model, val_loader, device, args.temperature)
            if val_loader is not None
            else float("nan")
        )
        epoch_metrics.append(
            {
                "epoch": epoch + 1,
                "train_avg_loss": train_avg,
                "val_avg_loss": val_avg,
            }
        )
        if val_loader is not None:
            print(
                f"epoch {epoch + 1} train_avg={train_avg:.4f} val_avg={val_avg:.4f} "
                f"(val split={args.val_split})"
            )
        else:
            print(f"epoch {epoch + 1} train_avg={train_avg:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    training_config = {
        "base_model": args.model_name,
        "query_prefix": args.query_prefix,
        "passage_prefix": args.passage_prefix,
        "max_length": args.max_length,
        "corpus_types": CORPUS_TYPES,
        "train_split": args.train_split,
        "val_split": args.val_split,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "temperature": args.temperature,
        "train_pairs": len(train_dataset),
        "epoch_metrics": epoch_metrics,
    }
    with open(os.path.join(args.output_dir, "retriever_config.json"), "w", encoding="utf-8") as file:
        json.dump(
            {
                "base_model": args.model_name,
                "query_prefix": args.query_prefix,
                "passage_prefix": args.passage_prefix,
                "max_length": args.max_length,
                "corpus_types": CORPUS_TYPES,
            },
            file,
            indent=2,
        )
    with open(os.path.join(args.output_dir, "training_metrics.json"), "w", encoding="utf-8") as file:
        json.dump(training_config, file, indent=2)
    print(f"Saved fine-tuned retriever to {args.output_dir}")
    if epoch_metrics:
        best = min(epoch_metrics, key=lambda row: row["val_avg_loss"] if row["val_avg_loss"] == row["val_avg_loss"] else float("inf"))
        if best["val_avg_loss"] == best["val_avg_loss"]:
            print(f"Best val epoch: {best['epoch']} (val_avg={best['val_avg_loss']:.4f})")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune dense retriever on TalkPlay conversations")
    parser.add_argument("--model-name", type=str, default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--output-dir", type=str, default="./checkpoints/bge-talkplay")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument(
        "--val-split",
        type=str,
        default="test",
        help="HF split for session-level validation (official dev set). Use '' to disable.",
    )
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument(
        "--query-prefix",
        type=str,
        default="Represent this sentence for searching relevant passages: ",
    )
    parser.add_argument("--passage-prefix", type=str, default="")
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    if parsed.val_split == "":
        parsed.val_split = None
    train(parsed)
