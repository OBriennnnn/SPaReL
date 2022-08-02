from typing import List
from utils import read_graph_file, dbpedia_sparql_query_process, get_candidate_relations, get_gold_relation, \
                  entity_process, standardization_lcquad_2, wikidata_sparql_query_process, get_wiki_candidate_relations, \
                  qald9_sparql_query_process, qald7_sparql_query_process, qald7_entity_process, qald7_relation_process, \
                  qald7_get_candidate_relations
import re
import json
from tqdm import tqdm

class GraphProcess:
    def __init__(self, graphs: List[list], data_file: str, data_set: str):

        self.graphs = graphs
        self.data_file = data_file
        self.data_set = data_set
        # self.data_sets

    @staticmethod
    def save_data(file_name, data):
        with open(file_name, 'w') as f:
            # json.dump(data, f)
            f.write(json.dumps(data, sort_keys=True, indent=4, separators=(',', ':')))
        f.close()

    @staticmethod
    def _left_space_count(amr_line: str) -> int:

        end = 0
        while amr_line[end] == ' ':
            end += 1
        tab = amr_line.count(' ', 0, end)
        return tab

    @staticmethod
    def _create_prior(amr_graph_with_prior: List[dict]) -> List[dict]:

        for amr_line_with_prior in amr_graph_with_prior:
            if ':polarity -' in amr_line_with_prior['amr_line']:
                amr_graph_with_prior.remove(amr_line_with_prior)

        pattern_name = re.compile(r'\:name \(z\d+ \/ name')
        for amr_line_with_prior in amr_graph_with_prior:
            if re.match(pattern_name, amr_line_with_prior['amr_line']) is not None:
                amr_graph_with_prior.remove(amr_line_with_prior)
        # print(amr_graph_with_prior)

        amr_graph_with_prior_length = len(amr_graph_with_prior)
        for idx in range(amr_graph_with_prior_length):
            if 'wiki' in amr_graph_with_prior[idx]['amr_line']:
                amr_graph_with_prior[idx-1]['amr_line'] = amr_graph_with_prior[idx-1]['amr_line'] \
                                                          + ' ' + amr_graph_with_prior[idx]['amr_line']
        # print(amr_graph_with_prior)

        pattern_wiki = re.compile(r'\:wiki')
        for amr_line_with_prior in amr_graph_with_prior:
            if re.match(pattern_wiki, amr_line_with_prior['amr_line']) is not None:
                amr_graph_with_prior.remove(amr_line_with_prior)
        # print(amr_graph_with_prior)

        amr_graph_with_prior_length = len(amr_graph_with_prior)
        for idx in range(amr_graph_with_prior_length):
            if ':wiki -' in amr_graph_with_prior[idx]['amr_line']:
                for j in range(idx+1, amr_graph_with_prior_length):
                    if ('op' in amr_graph_with_prior[j]['amr_line']) and ('wiki' not in amr_graph_with_prior[j]['amr_line']):
                        amr_graph_with_prior[idx]['amr_line'] = amr_graph_with_prior[idx]['amr_line'] \
                                                                    + ' ' + amr_graph_with_prior[j]['amr_line']
                    else:
                        break
        # print(amr_graph_with_prior)
        for amr_line_with_prior in amr_graph_with_prior:
            if ':wiki -' in amr_line_with_prior['amr_line']:
                amr_line = amr_line_with_prior['amr_line'].split()
                temp_index = amr_line.index(':wiki')
                temp_amr_line = amr_line[temp_index+2:]
                temp_amr_line = list(filter(lambda x: ':op' not in x, temp_amr_line))
                amr_line = amr_line[:temp_index + 2] + temp_amr_line
                amr_line_with_prior['amr_line'] = " ".join(amr_line)

        pattern_op = re.compile(r'\:op\d+ \"')
        for amr_line_with_prior in amr_graph_with_prior:
            if re.match(pattern_op, amr_line_with_prior['amr_line']) is not None:
                amr_graph_with_prior.remove(amr_line_with_prior)

        for amr_line_with_prior in amr_graph_with_prior:
            if re.match(pattern_op, amr_line_with_prior['amr_line']) is not None:
                amr_graph_with_prior.remove(amr_line_with_prior)
        # print(amr_graph_with_prior)
        new_amr_graph_with_prior = []
        amr_graph_with_prior_length = len(amr_graph_with_prior)
        for idx in range(amr_graph_with_prior_length):
            if amr_graph_with_prior[idx]['prior'] == 0:
                new_amr_graph_with_prior.append({'prior': -1, 'amr_line': amr_graph_with_prior[idx]['amr_line']})
            else:
                for backward_idx in range(idx, -1, -1):
                    if amr_graph_with_prior[backward_idx]['prior'] < amr_graph_with_prior[idx]['prior']:
                        new_amr_graph_with_prior.append({'prior': backward_idx,
                                                         'amr_line': amr_graph_with_prior[idx]['amr_line']})
                        break
        return new_amr_graph_with_prior

    def _construct_amr_graph(self, amr_graph_with_prior: List[dict]) -> List[dict]:

        pattern_node = None
        if self.data_set == 'lcquad1':
            pattern_node = re.compile(r'\:[\w-]+ [^\-]')
        elif self.data_set == 'lcquad2':
            pattern_node = re.compile(r'\:[\w-]+ [^\*]')
        elif self.data_set == 'qald9':
            pattern_node = re.compile(r'\:[\w-]+ [^\-]')
        elif self.data_set == 'qald7':
            pattern_node = re.compile(r'\:[\w-]+ [^\-]')

        for amr_dic in amr_graph_with_prior:
            amr_dic['amr_line'] = amr_dic['amr_line'].rstrip(')')
        amr_graph = []
        for amr_dic in amr_graph_with_prior:
            if amr_dic['prior'] == -1:
                amr_dic['amr_line'] = amr_dic['amr_line'].lstrip('(')
                amr_graph.append({'prior': -1, 'prior_node': None, 'edge': None, 'node': amr_dic['amr_line']})
            else:
                cut_index = re.match(pattern_node, amr_dic['amr_line'])

                edge = amr_dic['amr_line'][cut_index.span()[0]:cut_index.span()[1]]
                edge = edge.rstrip('(')
                edge = edge.rstrip(' ')
                node = amr_dic['amr_line'].lstrip(edge)
                node = node.lstrip(' ')
                node = node.lstrip('(')
                amr_graph.append({'prior': amr_dic['prior'], 'prior_node': amr_graph[amr_dic['prior']]['node'],
                                  'edge': edge, 'node': node})
                for amr_dic in amr_graph:
                    if len(amr_dic['node']) <= 3:
                        pattern_pron = amr_dic['node'] + ' /'
                        for amr_dic1 in amr_graph:
                            if pattern_pron in amr_dic1['node']:
                                amr_dic['node'] = amr_dic1['node']
        return amr_graph

    def construct_graph(self):
        amr_graphs = []
        for graph in self.graphs:
            question = graph[1]
            amr_graph = graph[2:]
            amr_graph_with_prior = [] # List[(tuple)]
            for amr_line in amr_graph:
                tab = self._left_space_count(amr_line)
                amr_graph_with_prior.append({'prior': tab, 'amr_line': amr_line.lstrip()})
            amr_graph_with_prior = self._create_prior(amr_graph_with_prior)
            amr_graph = self._construct_amr_graph(amr_graph_with_prior)
            amr_graphs.append({'question': question, 'amr_graph': amr_graph})
            # amr_graphs.append(amr_graph)
        return amr_graphs

    def _get_shortest_path(self, start_node: str, end_node: str, amr_graph: List[dict]) -> List[str]:
        """
        :param start_node: amr-unknown
        :param end_node: mention
        :param amr_graph: wholly graph
        :return: path
        """

        nodes = []
        nodes_neighbours = []
        for amr_dic in amr_graph:
            if amr_dic['node'] not in nodes:
                nodes.append(amr_dic['node'])

        for node in nodes:
            node_neighbours = set()
            for amr_dic in amr_graph:
                if node == amr_dic['node']:
                    if amr_dic['prior_node'] != None:
                        node_neighbours.add(amr_dic['prior_node'])
                elif node == amr_dic['prior_node']:
                    node_neighbours.add(amr_dic['node'])
            nodes_neighbours.append(node_neighbours)

        distance = []
        for i in range(len(nodes)):
            distance.append(100)
        distance[nodes.index(start_node)] = 0
        queue = []
        visited = []
        queue.insert(0, start_node)
        while len(queue) > 0:
            top = queue.pop()
            visited.append(top)
            if top != end_node:
                index = nodes.index(top)
                for neighbour_node in nodes_neighbours[index]:
                    if neighbour_node not in visited:
                        distance_index = nodes.index(neighbour_node)
                        distance[distance_index] = distance[index] + 1
                        queue.insert(0, neighbour_node)
            else:
                break

        stack = []
        stack.append(end_node)
        while stack[-1] != start_node:
            index = nodes.index(stack[-1])
            for neighbour_node in nodes_neighbours[index]:
                if distance[nodes.index(neighbour_node)] == distance[index] - 1:
                    for amr_dic in amr_graph:
                        if (amr_dic['node'] == nodes[nodes.index(neighbour_node)] and amr_dic['prior_node'] == stack[-1]) or \
                        (amr_dic['prior_node'] == nodes[nodes.index(neighbour_node)] and amr_dic['node'] == stack[-1]):
                            stack.append(amr_dic['edge'])
                            stack.append(nodes[nodes.index(neighbour_node)])
        amr_path = []
        for i in range(len(stack)):
            temp = stack.pop()
            amr_path.append(temp)

        for amr_dic in amr_graph:
            if amr_dic['prior_node'] is not None:
                if ('amr-unknown' in amr_dic['prior_node']) and (amr_dic['edge'] == ':mod' or amr_dic['edge'] == ':part-of'):
                    amr_path[0] = amr_dic['node']
        # if 'amr-unknown' in amr_path[0]:
        #     del amr_path[0]
        return amr_path

    def _get_slot(self, path: List[str]) -> dict:

        dic = {}
        mention = ''
        entity = ''
        candidate_entity = path[-1].split()
        for i in range(len(candidate_entity)):
            if candidate_entity[i] == ':wiki':
                if candidate_entity[i+1] == '-':
                    # 判断为mention
                    for j in range(i+2, len(candidate_entity)):
                        mention += ' ' + candidate_entity[j].strip("\"")
                else:
                    for j in range(i+1, len(candidate_entity)):
                        entity += candidate_entity[j].strip("\"")
        if mention != '':
            mention = mention.strip(" ")
            dic['mention'] = mention
        else:
            dic['entity'] = entity
        slot = []
        slot.append(path[-1])
        pattern = re.compile(r'\w+\-\d+')
        for i in range(len(path)-2, -1, -1):
            if ((pattern.search(path[i]) is not None) or ('have-rel-role-91' in path[i])) and (i != 1):
                slot.append(path[i+1])
                slot.append(path[i])
                slot.append(path[i-1])
                slot.append(path[i-2])
                break
        new_slot = []
        if len(slot) > 4:
            for s in slot:
                temp = s.split()
                if '/' in temp:
                    new_slot.append(temp[2])
                else:
                    new_slot.append(temp[0])
        else:
            if len(path) == 5:
                for p in path:
                    temp = p.split()
                    if '/' in temp:
                        new_slot.append(temp[2])
                    else:
                        new_slot.append(temp[0])
            elif len(path) > 5:
                if path[1] == ':mod':
                    temp_candidate_last_entity = path[-1].split()
                    for i in range(2, 6):
                        temp = path[i].split()
                        if '/' in temp:
                            new_slot.append(temp[2])
                        else:
                            new_slot.append(temp[0])
                    new_slot.append(temp_candidate_last_entity)
                else:
                    if len(path) == 7:
                        for i in range(2, 7):
                            temp = path[i].split()
                            if '/' in temp:
                                new_slot.append(temp[2])
                            else:
                                new_slot.append(temp[0])
                    else:
                        if len(path) == 9:
                            temp_candidate_last_entity = path[-1].split()
                            for i in range(2, 5):
                                temp = path[i].split()
                                if '/' in temp:
                                    new_slot.append(temp[2])
                                else:
                                    new_slot.append(temp[0])
                            if path[-2] == ':mod' or path[-2] == ':op1' or path[-2] == ':op2':
                                new_slot.append(path[5])
                                new_slot.append(temp_candidate_last_entity)
                            else:
                                new_slot.append(path[-2])
                                new_slot.append(temp_candidate_last_entity)
                        else: # len(path) > 11
                            temp_candidate_last_entity = path[-1].split()
                            for i in range(2, 6):
                                temp = path[i].split()
                                if '/' in temp:
                                    new_slot.append(temp[2])
                                else:
                                    new_slot.append(temp[0])
                            new_slot.append(temp_candidate_last_entity)
        dic['slot'] = new_slot

        return dic

    def slot_prediction(self, amr_graph):

        flag = 0
        start_node = ''
        for amr_dic in amr_graph:
            if 'amr-unknown' in amr_dic['node']:
                flag = 1
                start_node = amr_dic['node']
        if flag == 1:
            num_of_entities = 0
            entities = []
            for amr_dic in amr_graph:
                if ':wiki' in amr_dic['node']:
                    num_of_entities += 1
                    entities.append(amr_dic['node'])
            if num_of_entities != 0:
                slot = []
                for idx in range(num_of_entities):
                    end_node = entities[idx]
                    path = self._get_shortest_path(start_node, end_node, amr_graph)
                    if len(path) > 4:
                        slot.append(self._get_slot(path))
                return slot
        else:
            return ''

    def data_generation_lc_quad_1(self):

        pattern = re.compile(r'\{.+\}')
        data_set_dics = []
        amr_graphs = self.construct_graph()
        with open(self.data_file) as file:
            load_dicts = json.load(file)
            id = 0
            for i in tqdm(range(0, len(load_dicts))):
                question = load_dicts[i]['corrected_question']
                sparql_query = load_dicts[i]['sparql_query']
                link_slice = pattern.search(sparql_query).span()
                line = sparql_query[link_slice[0]: link_slice[1]]
                entity_relation_pairs = dbpedia_sparql_query_process(line)
                slot = self.slot_prediction(amr_graphs[i]['amr_graph'])
                if slot != [] and slot is not None:
                    for entity_relation_pair in entity_relation_pairs:
                        entity = entity_process(entity_relation_pair['entity'])
                        gold_relation = get_gold_relation(entity_relation_pair['gold_relation'])
                        candidate_relations = get_candidate_relations(entity_relation_pair['entity'])
                        candidate_relations.append(gold_relation)
                        candidate_relations = list(set(candidate_relations))
                        for i in range(len(candidate_relations)):
                            if 'wiki' in candidate_relations[i]:
                                candidate_relations[i] = 'wiki'
                        while 'wiki' in candidate_relations:
                            candidate_relations.remove('wiki')

                        gold_entity_lower = entity.lower()
                        entity_slot = []
                        if len(slot) > 1:
                            for s in slot:
                                if 'entity' in s.keys():
                                    if s['entity'].lower() == gold_entity_lower:
                                        entity_slot = s['slot']
                                if 'mention' in s.keys():
                                    temp = s['mention'].split()
                                    str_temp = "_".join(temp)
                                    if str_temp.lower() == gold_entity_lower:
                                        entity_slot = s['slot']
                                data_set_dic = {'id': id, 'question': question, 'entity': entity,
                                                'gold_relation': gold_relation,
                                                'candidate_relations': candidate_relations, 'slot': entity_slot}
                        elif len(slot) == 1:
                            entity_slot = slot[0]['slot']
                            data_set_dic = {'id': id, 'question': question, 'entity': entity,
                                            'gold_relation': gold_relation,
                                            'candidate_relations': candidate_relations, 'slot': entity_slot}

                        if data_set_dic['slot'] != []:
                            print(data_set_dic)
                            data_set_dics.append(data_set_dic)
                            id += 1
        print(len(data_set_dics))
        return data_set_dics


    def data_generation_lc_quad_2(self):
        pattern = re.compile(r'\{.+\}')
        data_set_dics = []
        amr_graphs = self.construct_graph()
        with open(self.data_file) as file:
            load_dicts = json.load(file)
            # id = 0
            for i in tqdm(range(0, len(amr_graphs))):
                question = load_dicts[i]['question']
                sparql_query = load_dicts[i]['sparql_wikidata']
                link_slice = pattern.search(sparql_query).span()
                line = sparql_query[link_slice[0]: link_slice[1]]
                entity_relation_pairs = wikidata_sparql_query_process(line)
                slot = self.slot_prediction(amr_graphs[i]['amr_graph'])
                if slot != [] and slot is not None:
                    if len(entity_relation_pairs) != 0:
                        for entity_relation_pair in entity_relation_pairs:
                            entity = entity_relation_pair['entity']
                            gold_relation = entity_relation_pair['gold_relation']
                            if entity != '' and gold_relation != '':
                                candidate_relations = get_wiki_candidate_relations(entity_relation_pair['entity_label'])
                                candidate_relations.append(gold_relation)
                                candidate_relations = list(set(candidate_relations))
                                gold_entity_lower = entity.lower()
                                entity_slot = []
                                if len(slot) > 1:
                                    for s in slot:
                                        if 'entity' in s.keys():
                                            if s['entity'].lower() == gold_entity_lower:
                                                entity_slot = s['slot']
                                        if 'mention' in s.keys():
                                            temp = s['mention'].split()
                                            str_temp = "_".join(temp)
                                            if str_temp.lower() == gold_entity_lower:
                                                entity_slot = s['slot']
                                        data_set_dic = {
                                            # 'id': id,
                                                        'question': question, 'entity': entity,
                                                        'gold_relation': gold_relation,
                                                        'candidate_relations': candidate_relations, 'slot': entity_slot}
                                elif len(slot) == 1:
                                    entity_slot = slot[0]['slot']
                                    data_set_dic = {
                                        # 'id': id,
                                                    'question': question, 'entity': entity,
                                                    'gold_relation': gold_relation,
                                                    'candidate_relations': candidate_relations, 'slot': entity_slot}

                                if data_set_dic['slot'] != []:
                                    with open('lcquad2_test_data.json', 'a', encoding='utf-8') as file_data:
                                        print(data_set_dic)
                                        json.dump(data_set_dic, file_data)
                                        file_data.write('\n')
            return data_set_dics

    def data_generation_qald_9(self):
        pattern = re.compile(r'\{.+\}')
        data_set_dics = []
        amr_graphs = self.construct_graph()
        with open(self.data_file) as file:
            load_dicts = json.load(file)
            for i in tqdm(range(0, len(amr_graphs))):
                # 从训练集的sparql语句中得到gold实体和句子
                question = load_dicts[i]['question']
                sparql_query = load_dicts[i]['sparql_query']
                link_slice = pattern.search(sparql_query).span()
                line = sparql_query[link_slice[0]: link_slice[1]]
                entity_relation_pairs = qald9_sparql_query_process(line)
                # print(entity_relation_pairs)
                slot = self.slot_prediction(amr_graphs[i]['amr_graph'])
                if slot != [] and slot is not None:
                    for entity_relation_pair in entity_relation_pairs:
                        entity = entity_process(entity_relation_pair['entity'])
                        gold_relation = entity_relation_pair['gold_relation']
                        candidate_relations = get_candidate_relations(entity_relation_pair['entity'])
                        candidate_relations.append(gold_relation)
                        candidate_relations = list(set(candidate_relations))
                        for i in range(len(candidate_relations)):
                            if 'wiki' in candidate_relations[i]:
                                candidate_relations[i] = 'wiki'
                        while 'wiki' in candidate_relations:
                            candidate_relations.remove('wiki')

                        gold_entity_lower = entity.lower()
                        entity_slot = []
                        if len(slot) > 1:
                            for s in slot:
                                if 'entity' in s.keys():
                                    if s['entity'].lower() == gold_entity_lower:
                                        entity_slot = s['slot']
                                if 'mention' in s.keys():
                                    temp = s['mention'].split()
                                    str_temp = "_".join(temp)
                                    if str_temp.lower() == gold_entity_lower:
                                        entity_slot = s['slot']
                                data_set_dic = {'question': question, 'entity': entity,
                                                'gold_relation': gold_relation,
                                                'candidate_relations': candidate_relations, 'slot': entity_slot}
                                if data_set_dic['slot'] != []:
                                    with open('qald9_test_data.json', 'a', encoding='utf-8') as file_data:
                                        print(data_set_dic)
                                        json.dump(data_set_dic, file_data)
                                        file_data.write('\n')
                                    file_data.close()
                        elif len(slot) == 1:
                            entity_slot = slot[0]['slot']
                            data_set_dic = {'question': question, 'entity': entity,
                                            'gold_relation': gold_relation,
                                            'candidate_relations': candidate_relations, 'slot': entity_slot}
                            if data_set_dic['slot'] != []:
                                with open('qald9_test_data.json', 'a', encoding='utf-8') as file_data:
                                    print(data_set_dic)
                                    json.dump(data_set_dic, file_data)
                                    file_data.write('\n')
                                file_data.close()

        return data_set_dics

    def data_generation_qald_7(self):
        pattern = re.compile(r'\{.+\}')
        data_set_dics = []
        amr_graphs = self.construct_graph()
        with open(self.data_file, encoding='utf-8') as file:
            load_dicts = json.load(file)
            for i in tqdm(range(0, len(amr_graphs))):
                question = load_dicts[i]['question']
                print(question)
                sparql_query = load_dicts[i]['query']
                link_slice = pattern.search(sparql_query).span()
                line = sparql_query[link_slice[0]: link_slice[1]]
                entity_relation_pairs = qald7_sparql_query_process(line)
                slot = self.slot_prediction(amr_graphs[i]['amr_graph'])
                # print(slot)
                if slot != [] and slot is not None and entity_relation_pairs is not None:
                    for entity_relation_pair in entity_relation_pairs:
                        entity = qald7_entity_process(entity_relation_pair['entity'])
                        gold_relation = qald7_relation_process(entity_relation_pair['gold_relation'])
                        candidate_relations = qald7_get_candidate_relations(entity_relation_pair['entity'])
                        candidate_relations.append(gold_relation)
                        candidate_relations = list(set(candidate_relations))
                        for i in range(len(candidate_relations)):
                            if 'wiki' in candidate_relations[i]:
                                candidate_relations[i] = 'wiki'
                        while 'wiki' in candidate_relations:
                            candidate_relations.remove('wiki')

                        gold_entity_lower = entity.lower()
                        entity_slot = []
                        if len(slot) > 1:
                            for s in slot:
                                if 'entity' in s.keys():
                                    if s['entity'].lower() == gold_entity_lower:
                                        entity_slot = s['slot']
                                if 'mention' in s.keys():
                                    temp = s['mention'].split()
                                    str_temp = "_".join(temp)
                                    if str_temp.lower() == gold_entity_lower:
                                        entity_slot = s['slot']
                                data_set_dic = {
                                    # 'id': id,
                                    'question': question, 'entity': entity,
                                    'gold_relation': gold_relation,
                                    'candidate_relations': candidate_relations, 'slot': entity_slot}
                                if data_set_dic['slot'] != []:
                                    data_set_dics.append(data_set_dic)
                        elif len(slot) == 1:
                            entity_slot = slot[0]['slot']
                            data_set_dic = {
                                # 'id': id,
                                'question': question, 'entity': entity,
                                'gold_relation': gold_relation,
                                'candidate_relations': candidate_relations, 'slot': entity_slot}

                            if data_set_dic['slot'] != []:
                                data_set_dics.append(data_set_dic)
        with open('sparl_qald7_test_data.json', 'w', encoding='utf-8') as file_data:
            json.dump(data_set_dics, file_data, indent=4, sort_keys=True)
        return data_set_dics





