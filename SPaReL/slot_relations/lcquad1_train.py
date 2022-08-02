import torch
import torch.nn as nn
from transformers import AutoTokenizer
from cross_encoder import CEM
from my_dataloader import MySLPDataLoader
from utils import slot_relation_pair_generation, get_top_k_candidate_relations
from tqdm import tqdm


num_of_epoch = 50
length_of_slot_relation_pair = 50
num_of_top_k_relation = 5

mslpdl_lcquad1 = MySLPDataLoader(train_data_name='../datas/LC-QuAD-1.0/sparl_lcquad1_train_data.json',
                                 test_data_name='../datas/LC-QuAD-1.0/sparl_lcquad1_test_data.json')
lcquad1_train_datas, lcquad1_test_datas = mslpdl_lcquad1.data_generation(num_of_slot_negative_data=3, batch_size=50)

train_datas = lcquad1_train_datas
test_datas = lcquad1_test_datas

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=checkpoint, cache_dir=None,
                                          force_download=None)
tokenizer.add_special_tokens({"additional_special_tokens": ["[AMR]", "[TEXT]", "[REL]"]})
amr_relation_labels = ['have-org-role', 'amr-unknown', ':ARG0-of', ':ARG1-of', ':ARG2-of', 'ARG3-of', ':ARG4-of',
                       ':ARG5-of',
                       ':ARG0', ':ARG1', ':ARG2', 'ARG3', ':ARG4', ':ARG5', 'have-rel-role', ':accompanier', ':age',
                       ':beneficiary',
                       ':compared-to', ':concession', ':condition', ':consist-of'':degree', ':destination',
                       ':direction',
                       ':domain', ':duration'':example', ':extent', ':frequency', ':instrument', ':location',
                       ':manner',
                       ':medium', ':mod', ':mode', ':name', ':ord', ':part', ':path', ':polarity', ':poss',
                       ':purpose',
                       ':quant', ':scale', ':source', ':subevent', ':time', ':topic', ':unit', ':value',
                       ':calendar', ':century',
                       ':day', ':dayperiod', ':decade', ':era', ':month', ':quarter', ':season', ':timezone',
                       ':weekday', ':year', ':year2']
tokenizer.add_tokens(new_tokens=amr_relation_labels)
tokenizer_length = len(tokenizer)  # 是为了给model中的resize传参

model = CEM(vocab_size=tokenizer_length)
criteon = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model.to(device)
criteon.to(device)

for epoch in tqdm(range(num_of_epoch)):
    model.train()
    for train_data in train_datas:
        slot_relation_pairs = train_data['slot_relation_pairs']
        labels = train_data['labels']
        input_slot_relation_pairs = tokenizer(slot_relation_pairs, padding='max_length', max_length=50,
                                              return_tensors='pt', truncation=True).to(device)
        true_labels = torch.tensor(labels).to(torch.float).to(device)
        logits = model(input_slot_relation_pairs)
        loss = criteon(logits.T.squeeze(0), true_labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(epoch, 'loss:', loss.item())
    sched.step()

    model.eval()
    with torch.no_grad():
        correct = 0.0
        total = 0.0
        for test_data in test_datas:
            test_slot = test_data['slot']
            test_candidate_relations = test_data['candidate_relations']
            test_gold_relation = test_data['gold_relation']
            candidate_slot_relation_pairs = []
            for test_candidate_relation in test_candidate_relations:
                candidate_slot_relation_pairs.append(slot_relation_pair_generation(slot=test_slot,
                                                                                   relation=test_candidate_relation))
            input_candidate_slot_relation_pairs = tokenizer(candidate_slot_relation_pairs, padding='max_length',
                                                            max_length=length_of_slot_relation_pair,
                                                            return_tensors='pt', truncation=True).to(device)

            logits = model(input_candidate_slot_relation_pairs)
            top_k_candidate_relation = get_top_k_candidate_relations(logits, test_candidate_relations,
                                                                     num_of_relation=num_of_top_k_relation)
            if test_gold_relation in top_k_candidate_relation:
                correct += 1
            total += 1
        acc = correct / total
        print(epoch, 'acc: ', acc)
