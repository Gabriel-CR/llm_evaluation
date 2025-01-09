# data_processing.py
import pandas as pd
import json
from datasets import load_dataset
from langchain.prompts import PromptTemplate

class DataProcessor:
    def __init__(self):
        # TODO: Limitar o número de tokens de saída
        # TODO: Melhorar o prompt
        # TODO: Colocar exemplo antes para a LLM
        # TODO: Solicitar que destaque a alternativa colocando {A}
        # TODO: Brincar com os parâmetros da LLM
        self.template = """Answer the following multiple choice question by giving the most appropriate response. Answer should be one among [A, B, C, D, E]

        Question: {prompt}\n
        A) {a}\n
        B) {b}\n
        C) {c}\n
        D) {d}\n
        E) {e}\n

        Answer:"""

    def format_text(self, example):
        prompt = PromptTemplate(template=self.template, input_variables=['prompt', 'a', 'b', 'c', 'd', 'e'])

        text = prompt.format(prompt=example['prompt'],
                                a=example['A'],
                                b=example['B'],
                                c=example['C'],
                                d=example['D'],
                                e=example['E'])
        return {"text": text}

    def convert_parquet_to_csv(self, parquet_file: str, csv_file: str, engine: str = "pyarrow"):
        """
        Converte um arquivo Parquet (incluindo compactado) para CSV.

        Args:
            parquet_file (str): Caminho do arquivo Parquet de entrada.
            csv_file (str): Caminho do arquivo CSV de saída.
            engine (str): Motor para ler o Parquet ("pyarrow" ou "fastparquet").
        """
        try:
            # Lê o arquivo Parquet
            df = pd.read_parquet(parquet_file, engine=engine)
            
            # Salva o DataFrame como CSV
            df.to_csv(csv_file, index=False)
            print(f"Arquivo convertido com sucesso: {csv_file}")
        except Exception as e:
            print(f"Erro ao converter o arquivo: {e}")

    def convert_json_to_csv(self, json_file: str, csv_file: str):
        """
        Converte um arquivo JSON para CSV.

        Args:
            json_file (str): Caminho do arquivo JSON de entrada.
            csv_file (str): Caminho do arquivo CSV de saída.
        """
        try:
            # Lê o arquivo JSON
            with open(json_file, 'r', encoding='utf-8') as file:
                data = json.load(file)

            # Converte os dados em um DataFrame do pandas
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                df = pd.DataFrame([data])
            else:
                raise ValueError("O formato do JSON não é suportado. Certifique-se de que seja uma lista ou um dicionário.")

            # Salva o DataFrame como CSV
            df.to_csv(csv_file, index=False, encoding='utf-8')
            print(f"Arquivo convertido com sucesso: {csv_file}")
        except Exception as e:
            print(f"Erro ao converter o arquivo: {e}")

    def read_data(self, path, type='csv'):
        dataset = None

        if type == 'csv':
            dataset = load_dataset("csv", data_files=path)
        elif type == 'json':
            self.convert_json_to_csv(path, 'data/convert_json.csv')
            dataset = load_dataset("csv", data_files='data/convert_json.csv')
        elif type == 'parquet':
            self.convert_parquet_to_csv(path, 'data/convert_parquet.csv')
            dataset = load_dataset("csv", data_files='data/convert_parquet.csv')
        else:
            print('[ERROR] formato inválido')
            return
        
        dataset = dataset.map(self.format_text)

        return dataset

    def format_to_eval(self, data, to_for):
        """
        Formata os dados para avaliação, organizando em um formato padronizado com colunas específicas.

        Args:
            data (list): Lista de dicionários contendo os dados brutos.
            to_for (dict): Dicionário com as chaves: id, prompt, A, B, C, D, E, answer.
                        Os valores devem ser listas correspondentes aos dados.

        Returns:
            pd.DataFrame: DataFrame com as colunas: id, prompt, A, B, C, D, E, answer.
        """
        # Inicializa as listas dentro do dicionário 'to_for'
        for key in ['id', 'prompt', 'A', 'B', 'C', 'D', 'E', 'answer']:
            if key not in to_for:
                to_for[key] = []

        # Itera sobre os dados brutos e preenche 'to_for'
        for entry in data:
            to_for['id'].append(entry.get('id'))
            to_for['prompt'].append(entry.get('prompt'))
            to_for['A'].append(entry.get('A'))
            to_for['B'].append(entry.get('B'))
            to_for['C'].append(entry.get('C'))
            to_for['D'].append(entry.get('D'))
            to_for['E'].append(entry.get('E'))
            to_for['answer'].append(entry.get('answer'))

        # Converte o dicionário 'to_for' em um DataFrame
        formatted_data = pd.DataFrame(to_for)

        return formatted_data
    
    def format_enem_dataset(self, data, column_convert):
        """
        Formata um dataset do ENEM para o formato de avaliação com colunas padronizadas.

        Args:
            data (DatasetDict): DatasetDict carregado com o Hugging Face datasets.
            column_convert (dict): Dicionário com as colunas a serem convertidas e as expressões correspondentes.

        Returns:
            pd.DataFrame: DataFrame formatado com as colunas especificadas no `column_convert`.
        """
        # Converte o DatasetDict para pandas DataFrame
        df = data['train'].to_pandas()  # 'train' pode variar conforme o split do dataset

        # Inicializa o DataFrame formatado
        formatted_data = {}

        # Itera sobre o mapeamento de colunas e aplica as transformações
        for col, expr in column_convert.items():
            formatted_data[col] = df.apply(lambda row: eval(expr), axis=1)

        # Converte o dicionário formatado em um DataFrame
        formatted_df = pd.DataFrame(formatted_data)

        return formatted_df

if __name__ == "__main__":
    processor = DataProcessor()
    dataset = processor.read_data('data/enem-2022.csv', type='csv')
    column_convert = {
        'id': "row['id']",
        'prompt': "row['question']",
        'A': "row['alternatives'].split(',')[0]",
        'B': "row['alternatives'].split(',')[1]",
        'C': "row['alternatives'].split(',')[2]",
        'D': "row['alternatives'].split(',')[3]",
        'E': "row['alternatives'].split(',')[4]",
        'answer': "row['label']"
    }

    # Formata o dataset
    formatted_df = processor.format_enem_dataset(dataset, column_convert)

    # Salvar dado formatado
    formatted_df.to_csv('data/enem-2022-formatted.csv', index=False)

    print(formatted_df.head())

