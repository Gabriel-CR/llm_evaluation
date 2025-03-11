# data_processing.py
import csv
import os
import pandas as pd
import json
from datasets import load_dataset
from langchain.prompts import PromptTemplate


class DataProcessor:
    def __init__(self):
        self.template = """Answer the following multiple choice question by giving the most appropriate response. Answer should be one among [A, B, C, D, E]. Add the correct alternative at the beginning of the answer and enclose it in braces. Example: {{A}}\n

        Question: {prompt}\n
        A) {a}\n
        B) {b}\n
        C) {c}\n
        D) {d}\n
        E) {e}\n

        Answer:"""

    def format_text(self, example):
        prompt = PromptTemplate(
            template=self.template, input_variables=["prompt", "a", "b", "c", "d", "e"]
        )

        text = prompt.format(
            prompt=example["prompt"],
            a=example["A"],
            b=example["B"],
            c=example["C"],
            d=example["D"],
            e=example["E"],
        )

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
        Converte um arquivo JSON no formato fornecido para um arquivo CSV,
        filtrando questões onde `has_associated_images` é `false` e usando `options` como alternativas.

        Args:
            json_file (str): Caminho do arquivo JSON de entrada.
            csv_file (str): Caminho do arquivo CSV de saída.
        """
        try:
            # Ler o arquivo JSON
            with open(json_file, "r", encoding="utf-8") as file:
                data = json.load(file)

            # Preparar a lista para os dados no formato desejado
            csv_data = []
            for entry in data:
                # Preencher os dados para o CSV
                csv_data.append(
                    {
                        "id": entry["id"],
                        "prompt": entry["question"],
                        "A": (
                            entry["options"][0] if len(entry["options"]) > 0 else ""
                        ),
                        "B": (
                            entry["options"][1] if len(entry["options"]) > 1 else ""
                        ),
                        "C": (
                            entry["options"][2] if len(entry["options"]) > 2 else ""
                        ),
                        "D": (
                            entry["options"][3] if len(entry["options"]) > 3 else ""
                        ),
                        "E": (
                            entry["options"][4] if len(entry["options"]) > 4 else ""
                        ),
                        "answer": entry.get(
                            "label", ""
                        ).upper(),  # Converte o label para maiúsculo
                        "images": ",".join(entry.get("associated_images", [])),
                    }
                )

            # Criar um DataFrame do pandas
            df = pd.DataFrame(csv_data)

            # Salvar o DataFrame como um arquivo CSV
            df.to_csv(csv_file, index=False, encoding="utf-8")
            print(f"Arquivo CSV gerado com sucesso: {csv_file}")
        except Exception as e:
            print(f"Erro ao converter o arquivo JSON para CSV: {e}")

    def jsonl_to_csv(self, jsonl_path, csv_path):
        try:
            with open(jsonl_path, 'r', encoding='utf-8') as jsonl_file, open(csv_path, 'w', newline='', encoding='utf-8') as csv_file:
                fieldnames = ['id', 'prompt', 'A', 'B', 'C', 'D', 'E', 'answer', 'images']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()

                for line in jsonl_file:
                    data = json.loads(line.strip())
                    row = {
                        'id': data.get('id', ''),
                        'prompt': data.get('question', ''),
                        'A': data.get('alternatives', [''] * 5)[0],
                        'B': data.get('alternatives', [''] * 5)[1],
                        'C': data.get('alternatives', [''] * 5)[2],
                        'D': data.get('alternatives', [''] * 5)[3],
                        'E': data.get('alternatives', [''] * 5)[4],
                        'answer': data.get('label', ''),
                        'images': ','.join(data.get('figures', []))
                    }
                    writer.writerow(row)
            print(f"Arquivo CSV gerado com sucesso: {csv_path}")
        except Exception as e:
            print(f"Erro ao converter o arquivo JSONL para CSV: {e}")

    def read_data(self, path, type="csv"):
        dataset = None
        data_path = os.getenv("DATA_PATH")

        if type == "csv":
            dataset = load_dataset("csv", data_files=path)
        elif type == "json":
            self.convert_json_to_csv(path, f"{data_path}convert_json.csv")
            dataset = load_dataset("csv", data_files=f"{data_path}convert_json.csv")
        elif type == "parquet":
            self.convert_parquet_to_csv(path, f"{data_path}convert_parquet.csv")
            dataset = load_dataset("csv", data_files=f"{data_path}convert_parquet.csv")
        elif type == "jsonl":
            self.jsonl_to_csv(path, f"{data_path}convert_jsonl.csv")
            dataset = load_dataset("csv", data_files=f"{data_path}convert_jsonl.csv")
        else:
            print("[ERROR] formato inválido")
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
        for key in ["id", "prompt", "A", "B", "C", "D", "E", "answer"]:
            if key not in to_for:
                to_for[key] = []

        # Itera sobre os dados brutos e preenche 'to_for'
        for entry in data:
            to_for["id"].append(entry.get("id"))
            to_for["prompt"].append(entry.get("prompt"))
            to_for["A"].append(entry.get("A"))
            to_for["B"].append(entry.get("B"))
            to_for["C"].append(entry.get("C"))
            to_for["D"].append(entry.get("D"))
            to_for["E"].append(entry.get("E"))
            to_for["answer"].append(entry.get("answer"))

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
        df = data[
            "train"
        ].to_pandas()  # 'train' pode variar conforme o split do dataset

        # Inicializa o DataFrame formatado
        formatted_data = {}

        # Itera sobre o mapeamento de colunas e aplica as transformações
        for col, expr in column_convert.items():
            formatted_data[col] = df.apply(lambda row: eval(expr), axis=1)

        # Converte o dicionário formatado em um DataFrame
        formatted_df = pd.DataFrame(formatted_data)

        return formatted_df

    def get_first_1000_rows(self, input_file: str, output_file: str):
        """
        Lê as primeiras 1000 linhas de um arquivo CSV e salva em outro arquivo.

        Args:
            input_file (str): Caminho do arquivo CSV de entrada.
            output_file (str): Caminho do arquivo CSV de saída.
        """
        try:
            # Lê apenas as primeiras 1000 linhas do arquivo
            df = pd.read_csv(input_file, nrows=1000)

            # Salva o resultado em um novo arquivo
            df.to_csv(output_file, index=False, encoding="utf-8")
            print(f"As primeiras 1000 linhas foram salvas em {output_file}")
        except Exception as e:
            print(f"Erro ao processar o arquivo: {e}")


if __name__ == "__main__":
    data_process = DataProcessor()
    data_process.jsonl_to_csv("./data/2023.jsonl", './data/2023_images.csv')
    data_process.jsonl_to_csv("./data/2024.jsonl", './data/2024_images.csv')
    data_process.jsonl_to_csv("./data/2022.jsonl", './data/2022_images.csv')
