import json
from utils import read_graph_file
from GraphProcess import GraphProcess


class MyDataLoader:
    def __init__(self, data_path: str):
        """
        :param data_path: 填写绝对或者相对路径到达 ‘/datas/’ 即可
        如果还要添加新的数据集，由于数据集内部的结构不同，需要自己写读文件的代码
        例子：self.data = ‘../datas/’
        """
        self.data_path = data_path

    def get_data(self):
        """
        按行读取文件，存储到数组中
        :return: List
        """
        datas = []
        with open(self.data_path, encoding='utf-8') as file:
            for line in file.readlines():
                line = json.loads(line)
                datas.append(line)
        file.close()
        id = 0
        for data in datas:
            data['id'] = id
            id += 1
        print(len(datas))
        print(datas[15])
        with open('sparl_qald9_test_data.json', 'w', encoding='utf-8') as f:
            f.write(json.dumps(datas, sort_keys=True, indent=4, separators=(',', ':')))
        f.close()


def main():
    mdl = MyDataLoader('qald9_test_data.json')
    mdl.get_data()

if __name__ == '__main__':
    main()

