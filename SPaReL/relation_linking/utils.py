import random


def beam_relation(relations):
    for relation in relations:
        if len(relation) == 1:
            relations.remove(relation)
    for relation in relations:
        relation = relation.split()
        if len(relation) > 3:
            relations.remove(' '.join(relation))

    for relation in relations:
        relation = relation.split()
        for r in relation:
            if len(r) == 1:
                relations.remove(' '.join(relation))
                break
    return relations


def find_relation(slot, relation_dic):
    # new_slot = [slot[0]] + [slot[2]] + [slot[4]]
    new_slot = slot
    # if 'amr-unknown' in new_slot:
    #     new_slot.remove('amr-unknown')

    # slot_key0 = [slot[0]] + [slot[2]] + [slot[4]]
    # new_slot.append(' '.join(slot_key0))
    #
    # slot_key1 = [slot[0]] + [slot[4]] + [slot[2]]
    # new_slot.append(' '.join(slot_key1))
    #
    # slot_key2 = [slot[2]] + [slot[0]] + [slot[4]]
    # new_slot.append(' '.join(slot_key2))
    #
    # slot_key3 = [slot[2]] + [slot[4]] + [slot[0]]
    # new_slot.append(' '.join(slot_key3))
    #
    # slot_key4 = [slot[4]] + [slot[2]] + [slot[0]]
    # new_slot.append(' '.join(slot_key4))
    #
    # slot_key5 = [slot[4]] + [slot[0]] + [slot[2]]
    # new_slot.append(' '.join(slot_key5))

    relations = []
    for rdk in relation_dic.keys():
        flag = 0
        s_key = rdk.split()
        for ns in new_slot:
            if ns not in s_key:
                flag = 1
                break
        if flag == 0:
            relations += relation_dic[rdk]
    relations = list(set(relations))

    relations = reduce_process(relations)


    #
    # for s in new_slot:
    #     if s in relation_dic.keys():
    #         relations += relation_dic[s]
            # print(s)

    return relations


def get_top_k_candidate_relations(logits, test_candidate_relations, num_of_relation=10):
    new_logits = logits.T.squeeze(0)
    top_k_candidate_relations = []
    new_logits = new_logits.sort(0, descending=True)
    if len(test_candidate_relations) > num_of_relation:
        for i in range(num_of_relation):
            top_k_candidate_relations.append(test_candidate_relations[new_logits.indices[i]])
    else:
        top_k_candidate_relations = test_candidate_relations
    return top_k_candidate_relations


def reduce_process(relations):
    for relation in relations:
        if (relation + 's') in relations:
            relations.remove(relation + 's')

    for relation in relations:
        if len(relation.split()) > 1:
            temp_relation = ''.join(relation.split())
            if temp_relation in relations:
                relations.remove(temp_relation)

    return relations


def get_candidate_relations(gold_relation, candidate_relations, top_k=10):
    candidate_relations = beam_relation(candidate_relations)
    candidate_relations = candidate_relations[0: top_k]
    if gold_relation not in candidate_relations:
        candidate_relations.pop(-1)
        candidate_relations.append(gold_relation)
        random.shuffle(candidate_relations)
    return candidate_relations

def lcquad2_get_candidate_relations(gold_relation, candidate_relations, top_k=10):
    candidate_relations = candidate_relations[0: top_k]
    if gold_relation not in candidate_relations:
        candidate_relations.pop(-1)
        candidate_relations.append(gold_relation)
        random.shuffle(candidate_relations)
    return candidate_relations
