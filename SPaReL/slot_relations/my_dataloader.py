import json
import random
import re
from utils import slot_relation_pair_generation

class MySLPDataLoader:
    def __init__(self, train_data_name, test_data_name):
        self.train_datas = self._get_data(train_data_name)
        self.test_datas = self._get_data(test_data_name)

    def _get_data(self, file_name):
        with open(file_name, encoding='utf-8') as file:
            datas = json.load(file)
        file.close()
        return datas

    def data_generation(self, num_of_slot_negative_data, batch_size=10):
        slot_relations_train_datas = []
        slot_relations_test_datas = []
        for train_data in self.train_datas:
            slot = self.slot_generation(train_data['slot'])
            gold_relation = train_data['gold_relation']
            candidate_relations = train_data['candidate_relations']
            candidate_relations.remove(gold_relation)
            random.shuffle(candidate_relations)
            positive_data = slot_relation_pair_generation(slot, gold_relation)
            slot_relations_train_datas.append({'slot_relation_pair': positive_data, 'label': 1})
            if len(candidate_relations) > num_of_slot_negative_data:
                for i in range(num_of_slot_negative_data):
                    if len(candidate_relations[i]) > 2:
                        negative_data = slot_relation_pair_generation(slot, candidate_relations[i])
                        slot_relations_train_datas.append({'slot_relation_pair': negative_data, 'label': 0})
            else:
                for i in range(len(candidate_relations)):
                    if len(candidate_relations[i]) > 2:
                        negative_data = slot_relation_pair_generation(slot, candidate_relations[i])
                        slot_relations_train_datas.append({'slot_relation_pair': negative_data, 'label': 0})
        random.shuffle(slot_relations_train_datas)
        slot_relations_train_datas = self.batch(slot_relations_train_datas, batch_size=batch_size)

        for test_data in self.test_datas:
            slot = self.slot_generation(test_data['slot'])
            gold_relation = test_data['gold_relation']
            candidate_relations = test_data['candidate_relations']
            slot_relations_test_datas.append({'slot': slot, 'gold_relation': gold_relation,
                                              'candidate_relations': candidate_relations})

        return slot_relations_train_datas, slot_relations_test_datas

    def slot_generation(self, slot):
        pattern = re.compile(r'-\d+$')
        new_slot = []
        for s in slot:
            slice = re.search(pattern, s)
            if slice is not None:
                s = s[0: slice.span()[0]]
                new_slot.append(s)
            else:
                new_slot.append(s)
        amr = []
        amr.append(new_slot[2])
        amr.append(new_slot[1])
        amr.append(new_slot[0])
        amr.append(new_slot[3])
        amr.append(new_slot[4])

        return ' '.join(amr)

    def batch(self, data, batch_size=10):
        train_batch_datas = []
        batch_datas = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        batch_datas = batch_datas[0: -1]
        for batch_data in batch_datas:
            slot_relation_pairs = []
            labels = []
            for dic in batch_data:
                slot_relation_pairs.append(dic['slot_relation_pair'])
                labels.append(dic['label'])
            train_batch_datas.append({'slot_relation_pairs': slot_relation_pairs, 'labels': labels})
        return train_batch_datas

