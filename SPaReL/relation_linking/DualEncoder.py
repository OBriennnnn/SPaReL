import torch
import torch.nn as nn
from transformers import AutoModel


class DualEncoder(nn.Module):
    def __init__(self, vocab_size):
        super(DualEncoder, self).__init__()
        self.model_question = AutoModel.from_pretrained('bert-base-cased')
        self.model_question.resize_token_embeddings(new_num_tokens=vocab_size)
        self.model_relations = AutoModel.from_pretrained('bert-base-cased')
        self.model_relations.resize_token_embeddings(new_num_tokens=vocab_size)
        self.question_fflayer = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256)
        )
        self.relation_fflayer = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64)
        )
        self.cls_module = nn.Sequential(
            nn.Linear(320, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 31),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(31, 1)
        )


    def forward(self, question, relation):
        question_output = self.model_question(**question)
        relation_output = self.model_relations(**relation)
        question_embedding = question_output[1]
        relation_embedding = relation_output[1]
        question_embedding = self.question_fflayer(question_embedding)
        question_embedding = question_embedding.repeat(relation_output[1].size(0), 1)
        relation_embedding = self.relation_fflayer(relation_embedding)
        cat_embedding = torch.cat((question_embedding, relation_embedding), dim=1)
        cat_embedding = self.cls_module(cat_embedding).T
        logits = cat_embedding.softmax(dim=1)
        return logits
