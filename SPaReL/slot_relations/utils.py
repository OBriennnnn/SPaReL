def slot_relation_pair_generation(slot, relation):
    slot_relation_data = ['[AMR]']
    slot_relation_data.append(slot)
    slot_relation_data.append('[REL]')
    slot_relation_data.append(relation)

    return ' '.join(slot_relation_data)

def get_top_k_candidate_relations(logits, test_candidate_relations, num_of_relation=10):
    top_k_candidate_relations = []
    new_logits = logits.T
    new_logits = new_logits.sort(-1, descending=True)
    if len(new_logits.indices[0]) > num_of_relation:
        for i in range(num_of_relation):
            top_k_candidate_relations.append(test_candidate_relations[new_logits.indices[0][i]])
    else:
        top_k_candidate_relations = test_candidate_relations
    return top_k_candidate_relations