from typing import List
import re
from SPARQLWrapper import SPARQLWrapper, SPARQLWrapper2, JSON
import sys


def return_sparql_query_results(query):
    user_agent = "WDQS-example Python/%s.%s" % (sys.version_info[0], sys.version_info[1])
    endpoint_url = "https://query.wikidata.org/sparql"
    # TODO adjust user agent; see https://w.wiki/CX6
    sparql = SPARQLWrapper(endpoint_url, agent=user_agent)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()


def read_graph_file(amr_path: str):
    """
    :param amr_path: 存储的amr图的文件
    :return: 返回存储了所有图结构的列表
    """
    graphs = []  # 存入所有的图
    with open(amr_path, encoding='utf-8') as file:
        line = file.readline()
        while line != 'The end!':
            line = file.readline()
            graph = []
            while line != '\n' and line != 'The end!':
                if line != '# ::status ParsedStatus.OK\n':
                    graph.append(line.rstrip('\n'))
                line = file.readline()
            graphs.append(graph)
    file.close()
    return graphs


def standardization_lcquad_2(sentence: str):
    sentence = sentence.split()
    last_word = sentence[-1]
    sentence = sentence[0:-1]
    if '?' in last_word:
        if '?' == last_word:
            last_word = ''
        else:
            last_word = last_word[0:-1]
            sentence.append(last_word)
    sentence.append('?')
    for i in range(len(sentence)):
        sentence[i] = sentence[i].strip('\"')
    while ' ' in sentence:
        del sentence[sentence.index(' ')]

    sentence = " ".join(sentence)
    sentence = "\"" + sentence + "\""
    return sentence


def dbpedia_sparql_query_process(sparql_query: str) -> List:
    sparql_query = sparql_query.lstrip("{")
    sparql_query = sparql_query.rstrip("}")
    sparql_query = sparql_query.strip()
    sparql_query = sparql_query.split()
    for s in sparql_query:
        if s == '.':
            sparql_query.remove(s)
    entity_relation_pairs = []
    for i in range(0, len(sparql_query), 3):
        slice_pair = sparql_query[i:i + 3]
        gold_relation = slice_pair[1]
        gold_relation = gold_relation.lstrip('<')
        gold_relation = gold_relation.rstrip('>')
        entity = ''
        if len(slice_pair[0]) > 20:
            entity = slice_pair[0]
        elif len(slice_pair[2]) > 20:
            entity = slice_pair[2]
        if entity != '' and gold_relation != 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type':
            entity_relation_pairs.append({'entity': entity, 'gold_relation': gold_relation})

    return entity_relation_pairs


def wikidata_sparql_query_process(sparql_query: str) -> List:
    sparql_query = sparql_query.lstrip("{")
    sparql_query = sparql_query.rstrip("}")
    sparql_query = sparql_query.strip()
    sparql_query = sparql_query.split('.')
    candidate_entity_relation_pairs = []
    for s in sparql_query:
        if len(s.split()) == 3:
            s = s.split()
            if 'wd:' in s[0] or 'wd:' in s[2]:
                if ':' in s[1]:
                    candidate_entity_relation_pairs.append(s)
    entity_relation_pairs = []
    for cerp in candidate_entity_relation_pairs:
        candidate_entity_label = ""
        if 'wd:' in cerp[0]:
            candidate_entity_label = cerp[0]
        elif 'wd:' in cerp[2]:
            candidate_entity_label = cerp[2]
        candidate_relation_label = cerp[1]
        entity = get_wiki_entity(candidate_entity_label)
        gold_relation = get_wiki_relation(candidate_relation_label)
        if entity != '':
            entity_relation_pairs.append({'entity': entity, 'entity_label': candidate_entity_label,
                                      'gold_relation': gold_relation})

    return entity_relation_pairs


def qald9_sparql_query_process(sparql_query: str) -> List:
    sparql_query = sparql_query.lstrip("{")
    sparql_query = sparql_query.rstrip("}")
    sparql_query = sparql_query.strip()
    sparql_query = sparql_query.split(';')
    entity_relation_pairs = []
    for s in sparql_query:
        if len(s.split()) == 3:
            s = s.split()
            if (s[1] != 'a') and ('rdf:' not in s[1]):
                if '<http://dbpedia.org/ontology/' in s[1] or '<http://dbpedia.org/property/' in s[1]:
                    gold_relation = process_relation(s[1][29:-1])
                    entity1 = get_dbpedia_entity(s[0])
                    entity2 = get_dbpedia_entity(s[2])
                    if entity1 != '' and entity2 == '':
                        entity_relation_pairs.append({'entity': entity1, 'gold_relation': gold_relation})
                    elif entity2 != '' and entity1 == '':
                        entity_relation_pairs.append({'entity': entity2, 'gold_relation': gold_relation})
                elif 'dbo:' in s[1] or 'dbp:' in s[1] or 'dct' in s[1]:
                    gold_relation = process_relation(s[1][4:])
                    entity1 = get_dbpedia_entity(s[0])
                    entity2 = get_dbpedia_entity(s[2])
                    if entity1 != '' and entity2 == '':
                        entity_relation_pairs.append({'entity': entity1, 'gold_relation': gold_relation})
                    elif entity2 != '' and entity1 == '':
                        entity_relation_pairs.append({'entity': entity2, 'gold_relation': gold_relation})
                elif 'onto:' in s[1] or 'foaf:' in s[1]:
                    gold_relation = process_relation(s[1][5:])
                    entity1 = get_dbpedia_entity(s[0])
                    entity2 = get_dbpedia_entity(s[2])
                    if entity1 != '' and entity2 == '':
                        entity_relation_pairs.append({'entity': entity1, 'gold_relation': gold_relation})
                    elif entity2 != '' and entity1 == '':
                        entity_relation_pairs.append({'entity': entity2, 'gold_relation': gold_relation})

    return entity_relation_pairs

def qald7_sparql_query_process(sparql_query: str) -> List:
    entity_relation_candidate_pairs = []
    entity_relation_candidate_pairs_link = []
    sparql_query = sparql_query.lstrip("{")
    sparql_query = sparql_query.rstrip("}")
    sparql_query = sparql_query.strip()
    sparql_query = sparql_query.split(' ')
    new_sparql_query = []
    for i in range(len(sparql_query)):
        if '>/<' in sparql_query[i]:
            q = sparql_query[i].split('>/<')
            sparql_query[i] = q[0] + '>'
    for i in range(len(sparql_query)):
        if ('?' in sparql_query[i] and '(?' not in sparql_query[i]) or '.' in sparql_query[i]:
            new_sparql_query.append(sparql_query[i])
    new_sparql_query = ' '.join(new_sparql_query)
    if '*' in new_sparql_query:
        new_sparql_query = []
    else:
        new_sparql_query = new_sparql_query.rstrip('.')
        new_sparql_query = new_sparql_query.rstrip(' ')
        new_sparql_query = new_sparql_query.split(' . ')
        new_sparql_query = ' '.join(new_sparql_query)
        new_sparql_query = new_sparql_query.split(' ')
        for i in range(int(len(new_sparql_query)/3)):
            entity_relation_candidate_pairs.append([new_sparql_query[3*i], new_sparql_query[3*i+1], new_sparql_query[3*i+2]])
        for entity_relation_candidate_pair in entity_relation_candidate_pairs:
            entity = ''
            gold_relation = entity_relation_candidate_pair[1]
            if 'http://www.wikidata.org/entity/' in entity_relation_candidate_pair[0]:
                entity = entity_relation_candidate_pair[0]
            elif 'http://www.wikidata.org/entity/' in entity_relation_candidate_pair[2]:
                entity = entity_relation_candidate_pair[2]
            if entity != '' and 'http://www.wikidata.org/' in gold_relation:
                entity = entity[32:-1]
            if 'http://www.wikidata.org/prop/' in gold_relation and 'http://www.wikidata.org/prop/direct/' \
                not in gold_relation and 'http://www.wikidata.org/prop/statement/' not in gold_relation:
                gold_relation = gold_relation[30:-1]
            elif 'http://www.wikidata.org/prop/qualifier/' in gold_relation:
                gold_relation = gold_relation[40:-1]
            elif 'http://www.wikidata.org/prop/statement/' in gold_relation:
                gold_relation = gold_relation[40:-1]
            elif 'http://www.wikidata.org/prop/direct/' in gold_relation:
                gold_relation = gold_relation[37:-1]
            else:
                gold_relation = ''

            if entity != '' and gold_relation != '':
                entity_relation_candidate_pairs_link.append({'entity': entity, 'gold_relation': gold_relation})
        # print(entity_relation_candidate_pairs_link)
        return entity_relation_candidate_pairs_link

    # print(org_sparql_query)



def get_dbpedia_entity(s: str):
    if '?' in s:
        return ''
    else:
        if 'res:' in s or 'dbr:' in s:
            s = '<http://dbpedia.org/resource/' + s[4:] + '>'
            return s
        elif 'dbo:' in s:
            s = '<http://dbpedia.org/ontology/' + s[4:] + '>'
            return s
        elif 'dbc:' in s:
            s = '<http://dbpedia.org/resource/Category:' + s[4:] + '>'
            return s
        elif 'foaf:' in s:
            s = '<http://xmlns.com/foaf/0.1/' + s[5:] + '>'
            return s
        elif '<http://dbpedia.org/resource/' in s or '<http://dbpedia.org/ontology/' in s:
            return s
        elif '<http://dbpedia.org/class/yago/' in s:
            return s
        else:
            return ''


def process_relation(s: str):
    pattern = "[A-Z]"
    s = re.sub(pattern, lambda x: " " + x.group(0), s)
    s = s.lower()
    return s


# def get_qald_candidate_relations(entity: str):
#     link = 'http://dbpedia.org/sparql'
#     sparql = SPARQLWrapper2(link)
#     # 组织一下查询语言
#     sparql_query = """
#             SELECT distinct ?relation
#             WHERE
#             {
#                 """ + entity + """ ?relation ?x
#             }
#         """
#     # print(sparql_query)
#     sparql.setQuery(sparql_query)
#     sparql.setReturnFormat(JSON)
#     relations = []
#     for result in sparql.query().bindings:
#         relations.append(result['relation'].value)
#
#     sparql_query = """
#                 SELECT distinct ?relation
#                 WHERE
#                 {
#                     ?x ?relation """ + entity + """
#                 }
#             """
#     sparql.setQuery(sparql_query)
#     sparql.setReturnFormat(JSON)
#     for result in sparql.query().bindings:
#         relations.append(result['relation'].value)
#     input_relations = encoder_relations(relations)
#     return input_relations


def get_wiki_entity(candidate_entity_label: str):
    entity = ''
    query_string = """
            SELECT DISTINCT * WHERE {
      """ + candidate_entity_label + """ rdfs:label ?label .
      FILTER (langMatches( lang(?label), "en" ) )
    }"""
    res = return_sparql_query_results(query_string)
    if len(res['results']['bindings']) > 0:
        entity = res['results']['bindings'][0]['label']['value']
    return entity


def get_wiki_relation(candidate_relation_label: str):
    relation = ""
    if 'p:' in candidate_relation_label:
        query_string = """
        SELECT ?pLabel WHERE {
          VALUES (?pp) {(""" + candidate_relation_label + """)}
          ?p wikibase:claim ?pp
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
        }
        """

        res = return_sparql_query_results(query_string)
        if len(res['results']['bindings']) > 0:
            relation = res['results']['bindings'][0]['pLabel']['value']
        # relation = res['results']['bindings'][0]['pLabel']['value']

    elif 'wdt:' in candidate_relation_label:
        query_string = """
          SELECT ?wdLabel WHERE {
          VALUES (?wdt) {(""" + candidate_relation_label + """)}
           ?wd wikibase:directClaim ?wdt .
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
        }"""

        res = return_sparql_query_results(query_string)
        if len(res['results']['bindings']) > 0:
            relation = res['results']['bindings'][0]['wdLabel']['value']
    elif 'ps:' in candidate_relation_label:
        query_string = """
        SELECT ?psLabel WHERE {
            VALUES(?pps) {(""" + candidate_relation_label + """)}
        ?ps wikibase:statementProperty ?pps.
        SERVICE wikibase:label {bd:serviceParam wikibase:language "en".}
        }"""

        res = return_sparql_query_results(query_string)
        if len(res['results']['bindings']) > 0:
            relation = res['results']['bindings'][0]['psLabel']['value']
    elif 'pq' in candidate_relation_label:
        query_string = """
               SELECT ?pqLabel WHERE {
                VALUES (?ppq) {(""" + candidate_relation_label + """)}
                ?pq  wikibase:qualifier ?ppq.
                SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
                }"""
        res = return_sparql_query_results(query_string)
        if len(res['results']['bindings']) > 0:
            relation = res['results']['bindings'][0]['pqLabel']['value']

    return relation


def get_wiki_candidate_relations(entity_label):
    relations = []
    query_string = """
    select distinct ?r ?rLabel {
    values (?subject) {(""" + entity_label + """)}
    ?subject ?relation ?item .
    ?r wikibase:directClaim ?relation .
    service wikibase:label { bd:serviceParam wikibase:language "en" }
}
    """
    res = return_sparql_query_results(query_string)
    candidate_relations = res['results']['bindings']
    for candidate_relation in candidate_relations:
        relation = candidate_relation['rLabel']['value']
        relations.append(relation)
    return relations


def get_candidate_relations(entity: str):
    link = 'http://dbpedia.org/sparql'
    sparql = SPARQLWrapper2(link)
    # 组织一下查询语言
    sparql_query = """
        SELECT distinct ?relation
        WHERE
        {
            """ + entity + """ ?relation ?x 
        }
    """
    # print(sparql_query)
    sparql.setQuery(sparql_query)
    sparql.setReturnFormat(JSON)
    relations = []
    for result in sparql.query().bindings:
        relations.append(result['relation'].value)

    sparql_query = """
            SELECT distinct ?relation
            WHERE
            {
                ?x ?relation """ + entity + """ 
            }
        """
    sparql.setQuery(sparql_query)
    sparql.setReturnFormat(JSON)
    for result in sparql.query().bindings:
        relations.append(result['relation'].value)
    input_relations = encoder_relations(relations)
    return input_relations


def encoder_relations(relations):
    new_relations = []
    for relation in relations:
        temp = relation_extraction(relation)
        if '#' not in temp:
            if not bool(re.search(r'\d', temp)):
                new_relations.append(temp)
    return new_relations


def relation_extraction(relation:str):
    pattern = re.compile(r'[^/]+(?!.*/)')
    relation_slice = pattern.search(relation).span()
    relation = relation[relation_slice[0]:relation_slice[1]]
    relation = re.sub(r"([A-Z])", r" \1", relation).split()
    for i in range(len(relation)-1):
        if relation[i] == 'I' and relation[i+1] == 'D':
            relation[i] = 'ID'
            relation.pop(i+1)
    relation = ' '.join(relation)
    relation = relation.lower()
    return relation


def get_gold_relation(relation: str):
    return relation_extraction(relation)


def entity_process(entity_link: str):
    pattern = re.compile(r'[^/]+(?!.*/)')
    entity_slice = pattern.search(entity_link).span()
    entity = entity_link[entity_slice[0]:entity_slice[1]]
    entity = entity.rstrip(">")
    return entity

def qald7_entity_process(candidate_entity_label):
    entity = ''
    query_string = """
            SELECT DISTINCT * WHERE {
       wd:""" + candidate_entity_label + """ rdfs:label ?label .
      FILTER (langMatches( lang(?label), "en" ) )
    }"""
    res = return_sparql_query_results(query_string)
    if len(res['results']['bindings']) > 0:
        entity = res['results']['bindings'][0]['label']['value']
    return entity

def qald7_relation_process(candidate_relation_label):
    relation = ""
    query_string = """
        SELECT ?pLabel WHERE {
          VALUES (?pp) {( p:""" + candidate_relation_label + """)}
          ?p wikibase:claim ?pp
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
        }
        """

    res = return_sparql_query_results(query_string)
    if len(res['results']['bindings']) > 0:
        relation = res['results']['bindings'][0]['pLabel']['value']
    return relation

def qald7_get_candidate_relations(entity_label):
    relations = []
    query_string = """
    select distinct ?r ?rLabel {
    values (?subject) {(wd:""" + entity_label + """)}
    ?subject ?relation ?item .
    ?r wikibase:directClaim ?relation .
    service wikibase:label { bd:serviceParam wikibase:language "en" }
}
    """
    res = return_sparql_query_results(query_string)
    candidate_relations = res['results']['bindings']
    for candidate_relation in candidate_relations:
        relation = candidate_relation['rLabel']['value']
        relations.append(relation)
    return relations

def remove_special_u(line):
    pattern = re.compile(r'\\.+\d')
    if pattern.search(line):
        slice = pattern.search(line).span()
        line = line[0: slice[0]] + line[slice[1]:]
    return line
