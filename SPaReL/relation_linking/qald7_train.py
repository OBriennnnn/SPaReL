import torch
from transformers import AutoTokenizer
from torch import nn
from qald7_dataloader import MyRLDataLoader
from tqdm import tqdm
from DualEncoder import DualEncoder
from slot_relations.cross_encoder import CEM
from utils import beam_relation, find_relation
from slot_relations.utils import slot_relation_pair_generation, get_top_k_candidate_relations

num_epochs = 50
slot_token_max_length = 15
question_token_max_length = 50
length_of_slot_relation_pair = 50
relation_token_max_length = 30
num_of_relation = 20

####载入数据####
mrldl = MyRLDataLoader(num_of_train_candidate_data=20)
train_datas = mrldl.get_train_data()
slot_relation_dic = mrldl.dic_generation()
test_datas= mrldl.get_test_data()

# 给tokenizer添加独立的特殊token
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=checkpoint, cache_dir=None, force_download=None)
tokenizer.add_special_tokens({"additional_special_tokens": ["[AMR]", "[TEXT]", "[REL]"]})
amr_relation_labels = ['have-org-role', 'amr-unknown', ':ARG0-of', ':ARG1-of', ':ARG2-of', 'ARG3-of', ':ARG4-of', ':ARG5-of',
    ':ARG0', ':ARG1', ':ARG2', 'ARG3', ':ARG4', ':ARG5', 'have-rel-role', ':accompanier', ':age', ':beneficiary',
    ':compared-to', ':concession', ':condition', ':consist-of'':degree', ':destination', ':direction',
    ':domain', ':duration'':example', ':extent', ':frequency', ':instrument', ':location', ':manner',
    ':medium', ':mod', ':mode', ':name', ':ord', ':part', ':path', ':polarity', ':poss', ':purpose',
    ':quant', ':scale', ':source', ':subevent', ':time', ':topic', ':unit', ':value', ':calendar', ':century',
    ':day', ':dayperiod', ':decade', ':era', ':month', ':quarter', ':season', ':timezone', ':weekday', ':year', ':year2']
tokenizer.add_tokens(new_tokens=amr_relation_labels)
tokenizer_length = len(tokenizer) #是为了给model中的resize传参

model = DualEncoder(vocab_size=tokenizer_length)
cem_model = CEM(vocab_size=tokenizer_length)
cem_model.load_state_dict(torch.load("../datas/qald7_cem_model.pkl"))
criteon = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-6)
sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model.to(device)
cem_model.to(device)
criteon.to(device)

for epoch in tqdm(range(num_epochs)):
    model.train()
    for train_data in train_datas:
        question = train_data['question']
        combine_question = train_data['combine_question']
        candidate_relations = train_data['candidate_relations']
        gold_relation = train_data['gold_relation']
        slot = train_data['slot']
        label = [candidate_relations.index(gold_relation)]
        input_question = tokenizer(question, padding='max_length', max_length=question_token_max_length,
                                   return_tensors='pt', truncation=True).to(device)
        input_combine_question = tokenizer(combine_question, padding='max_length', max_length=question_token_max_length,
                                           return_tensors='pt', truncation=True).to(device)
        input_relations = tokenizer(candidate_relations, padding='max_length', max_length=relation_token_max_length,
                                    return_tensors='pt', truncation=True).to(device)
        true_label = torch.tensor(label).to(torch.long).to(device)
        logits = model(input_combine_question, input_relations)
        loss = criteon(logits, true_label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(epoch, 'loss:', loss.item())
    sched.step()

    with torch.no_grad():
        model.eval()
        qald7_p = 0.0
        qald7_r = 0.0
        for test_data in test_datas:
            qald7_predictions = []
            qald7_golds = []
            question = test_data['question']
            combine_question = test_data['combine_question']
            candidate_relations = test_data['candidate_relations']
            gold_relation = test_data['gold_relation']
            slot = test_data['slot']
            dic_relations = find_relation(slot, slot_relation_dic)
            input_combine_question = tokenizer(combine_question, padding='max_length',
                                               max_length=question_token_max_length,
                                               return_tensors='pt', truncation=True).to(device)

            if len(dic_relations) != 0:
                input_slot = tokenizer(' '.join(slot), padding='max_length', max_length=slot_token_max_length,
                                       return_tensors='pt', truncation=True).to(device)
                input_candidate_relations = tokenizer(candidate_relations, padding='max_length',
                                                      max_length=relation_token_max_length,
                                                      return_tensors='pt', truncation=True).to(device)
                candidate_slot_relation_pairs = []
                for candidate_relation in candidate_relations:
                    candidate_slot_relation_pairs.append(slot_relation_pair_generation(slot=' '.join(slot),
                                                                                       relation=candidate_relation))
                input_candidate_slot_relation_pairs = tokenizer(candidate_slot_relation_pairs, padding='max_length',
                                                                max_length=length_of_slot_relation_pair,
                                                                return_tensors='pt', truncation=True).to(device)

                cem_logits = cem_model(input_candidate_slot_relation_pairs)
                reduced_relations = get_top_k_candidate_relations(cem_logits, candidate_relations,
                                                                  num_of_relation=num_of_relation)
                dic_relations += reduced_relations
                input_dic_relations = tokenizer(dic_relations, padding='max_length',
                                                max_length=relation_token_max_length,
                                                return_tensors='pt', truncation=True).to(device)
                slot_logits = model(input_combine_question, input_dic_relations)
                pred = slot_logits.argmax(dim=1)
                qald7_predictions.append(dic_relations[pred])
                qald7_golds.append(gold_relation)
            else:
                input_slot = tokenizer(' '.join(slot), padding='max_length', max_length=slot_token_max_length,
                                       return_tensors='pt', truncation=True).to(device)
                input_candidate_relations = tokenizer(candidate_relations, padding='max_length',
                                                      max_length=relation_token_max_length,
                                                      return_tensors='pt', truncation=True).to(device)
                candidate_slot_relation_pairs = []
                for candidate_relation in candidate_relations:
                    candidate_slot_relation_pairs.append(slot_relation_pair_generation(slot=' '.join(slot),
                                                                                       relation=candidate_relation))
                input_candidate_slot_relation_pairs = tokenizer(candidate_slot_relation_pairs, padding='max_length',
                                                                max_length=length_of_slot_relation_pair,
                                                                return_tensors='pt', truncation=True).to(device)

                cem_logits = cem_model(input_candidate_slot_relation_pairs)
                reduced_relations = get_top_k_candidate_relations(cem_logits, candidate_relations,
                                                                  num_of_relation=num_of_relation)
                input_reduced_relations = tokenizer(reduced_relations, padding='max_length',
                                                    max_length=relation_token_max_length,
                                                    return_tensors='pt', truncation=True).to(device)

                reduced_logits = model(input_combine_question, input_reduced_relations)
                reduced_logits = reduced_logits.to(device)
                pred = reduced_logits.argmax(dim=1)
                qald7_predictions.append(reduced_relations[pred])
                qald7_golds.append(gold_relation)

            qald7_p = (len(set(qald7_predictions) & set(qald7_golds))) / len(set(qald7_predictions))
            qald7_r = (len(set(qald7_predictions) & set(qald7_golds))) / len(set(qald7_golds))

    qald7_p /= len(test_datas)
    qald7_r /= len(test_datas)
    qald7_f1 = 2 * ((qald7_p * qald7_r) / (qald7_p + qald7_r))
    print(epoch, 'p: ', qald7_p, 'r: ', qald7_r, 'f1: ', qald7_f1)
