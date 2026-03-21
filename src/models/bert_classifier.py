"""
bert_classifier.py
------------------
GuardianBERT: BERT-based multi-label toxic content classifier.

Architecture:
    BERT (bert-base-uncased) → [CLS] pooler_output
    → Dropout(0.3) → Linear(768 → 256) → ReLU
    → Dropout(0.2) → Linear(256 → 4) → Sigmoid

    Output: 4 independent confidence scores (one per label).
    Multi-label: any combination of labels can be active simultaneously.

Labels: [normal, depressive, hate_speech, violent]
"""

import torch
import torch.nn as nn
from transformers import BertModel


class GuardianBERT(nn.Module):
    """
    Multi-label BERT classifier for toxic content detection.

    Args:
        num_labels (int): Number of output labels (default 4).
        dropout (float): Dropout probability after BERT pooler (default 0.3).
        model_name (str): HuggingFace BERT variant name.
    """

    def __init__(
        self,
        num_labels: int = 4,
        dropout: float = 0.3,
        model_name: str = "bert-base-uncased",
    ):
        super(GuardianBERT, self).__init__()
        self.num_labels = num_labels
        self.model_name = model_name

        # BERT backbone — frozen or fine-tuned depending on config
        self.bert = BertModel.from_pretrained(model_name)

        # Classification head
        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(768, 256)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, num_labels)
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: Tensor [batch_size, seq_len]
            attention_mask: Tensor [batch_size, seq_len]
            token_type_ids: Optional segment IDs

        Returns:
            Tensor [batch_size, num_labels] with sigmoid probabilities.
        """
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # Use [CLS] token pooler output (BERT's pooled representation)
        cls_output = bert_outputs.pooler_output  # [batch_size, 768]

        x = self.dropout1(cls_output)
        x = self.fc1(x)          # [batch_size, 256]
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)          # [batch_size, num_labels]
        return self.sigmoid(x)   # [batch_size, num_labels] — probabilities

    def freeze_bert(self) -> None:
        """Freeze BERT backbone (train only classification head)."""
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert(self) -> None:
        """Unfreeze BERT backbone for full fine-tuning."""
        for param in self.bert.parameters():
            param.requires_grad = True

    def count_parameters(self) -> int:
        """Returns total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns raw logits (before sigmoid) — used with BCEWithLogitsLoss.
        """
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        cls_output = bert_outputs.pooler_output
        x = self.dropout1(cls_output)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        return self.fc2(x)  # raw logits, no sigmoid
