import json


def read_file(file_name):
    with open(file_name) as f:
        data = json.load(f)
    return data['questions']

with open('train-data-QALD-7.json', 'w', encoding='utf-8') as file:
    datas = read_file('qald-7-train-en-wikidata.json')
    train_datas = []
    for i in range(len(datas)):
        question = datas[i]['question'][0]['string']
        p = question[-1]
        question = question[0:-1] + ' ' + p
        query = datas[i]['query']['sparql']
        train_datas.append({'id': i, 'question': question, 'query': query})
    json.dump(train_datas, file, indent=4, ensure_ascii=False)
    file.write('\n')
    file.close()

with open('test-data-QALD-7.json', 'w', encoding='utf-8') as file:
    datas = read_file('qald-7-test-en-wikidata.json')
    test_datas = []
    for i in range(len(datas)):
        question = datas[i]['question'][0]['string']
        p = question[-1]
        question = question[0:-1] + ' ' + p
        query = datas[i]['query']['sparql']
        test_datas.append({'id': i, 'question': question, 'query': query})
    json.dump(test_datas, file, indent=4, ensure_ascii=False)
    file.write('\n')
    file.close()