import torch
import torch.nn as nn
from transformers import AutoModel


class CEM(nn.Module):
    def __init__(self, vocab_size):
        super(CEM, self).__init__()
        self.model = AutoModel.from_pretrained('bert-base-cased')
        self.model.resize_token_embeddings(new_num_tokens=vocab_size)
        self.fflayer = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )

    def m_average(self, m, word_embedding):
        output, not_need = torch.split(word_embedding, split_size_or_sections=[m, word_embedding.size(1)-m], dim=1)
        output = output.mean(dim=1)
        return output

    def forward(self, input_slot_relation_pair):
        output = self.model(**input_slot_relation_pair)
        output_embedding = output[1]
        logits = self.fflayer(output_embedding)
        return logits.sigmoid()
