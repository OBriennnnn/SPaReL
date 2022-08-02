import json
import random
import re
from utils import beam_relation, lcquad2_get_candidate_relations

class MyRLDataLoader:
    def __init__(self, num_of_train_candidate_data=10):
        self.num_of_train_candidate_data = num_of_train_candidate_data

    def _get_data(self, file_name):
        with open(file_name, encoding='utf-8') as file:
            datas = json.load(file)
        file.close()
        return datas

    def get_train_data(self):
        rl_train_datas = []
        qald9_train_datas = self._get_data('../datas/QALD-7/sparl_qald7_train_data.json')
        self.train_datas = qald9_train_datas
        random.shuffle(self.train_datas)
        for i in range(len(self.train_datas)):
            if len(self.train_datas[i]['gold_relation'].split()) <= 3:
                question = self.train_datas[i]['question']
                slot = self.train_datas[i]['slot']
                gold_relation = self.train_datas[i]['gold_relation']
                slot, combine_question = self.question_combine(question, slot)
                candidate_relations = self.train_datas[i]['candidate_relations']
                # candidate_relations = beam_relation(candidate_relations)
                candidate_relations = lcquad2_get_candidate_relations(gold_relation, candidate_relations,
                                                              top_k=self.num_of_train_candidate_data)
                rl_train_datas.append({'question': question,
                                       'combine_question': combine_question,
                                       'candidate_relations': candidate_relations,
                                       'gold_relation': gold_relation,
                                       'slot': slot})
        return rl_train_datas

    def get_test_data(self):
        # qald9_test_datas = self._get_data('../datas/QALD-7/sparl_qald7_test_data.json') \
        #                    + self._get_data('../datas/QALD-7/sparl_qald7_train_data.json')
        qald9_test_datas = self._get_data('../datas/QALD-7/sparl_qald7_test_data.json')
        rl_test_datas = self.test_data_generation(qald9_test_datas)

        return rl_test_datas

    def test_data_generation(self, test_datas):
        rl_test_datas = []
        random.shuffle(test_datas)
        for i in range(len(test_datas)):
            if len(test_datas[i]['gold_relation'].split()) <= 3:
                question = test_datas[i]['question']
                slot = test_datas[i]['slot']
                slot, combine_question = self.question_combine(question, slot)
                candidate_relations = test_datas[i]['candidate_relations']
                # candidate_relations = beam_relation(candidate_relations)
                gold_relation = test_datas[i]['gold_relation']
                rl_test_datas.append({'question': question,
                                      'combine_question': combine_question,
                                      'candidate_relations': candidate_relations,
                                      'gold_relation': gold_relation,
                                      'slot': slot})
        return rl_test_datas

    def question_combine(self, question, slot):
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

        question_list = []
        question_list.append('[AMR]')
        question_list = question_list + amr
        question_list.append('[TEXT]')
        question_str = ' '.join(question_list)
        question = question_str + ' ' +question
        return amr, question

    def dic_generation(self):
        r_s_dic = {}
        for train_data in self.train_datas:
            gold_relation = train_data['gold_relation']
            slot, _ = self.question_combine(train_data['question'], train_data['slot'])
            slot = ' '.join(slot)
            if slot not in r_s_dic.keys():
                r_s_dic[slot] = [gold_relation]
            elif slot in r_s_dic.keys():
                r_s_dic[slot].append(gold_relation)

        for k in r_s_dic.keys():
            r_s_dic[k] = set(r_s_dic[k])
            r_s_dic[k] = list(r_s_dic[k])

        return r_s_dic
