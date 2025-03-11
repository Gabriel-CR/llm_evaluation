import base64
import datetime
import pandas as pd
from dotenv import load_dotenv
import csv
import re
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from data_processing import DataProcessor
from metrics import Metrics

class Evaluete:
    def __init__(self, dataset):
        self.metrics = Metrics()
        self.dataset = dataset
        load_dotenv()
        self.results_dir = os.getenv("RESULTS_PATH")
        self.interaction_dir = os.getenv("INTERACTION_PATH")

    def evaluate(self, model, method, file_name, model_name, max_tokens=5):
        if method == "accuracy":
            self.evaluate_acc(model, file_name, model_name, max_tokens)
        elif method == "metrics":
            self.evaluate_with_metrics(model, file_name, model_name, max_tokens)
        elif method == "apk":
            self.evaluate_apk(model, file_name, model_name, max_tokens)
        elif method == "accuracy_with_image":
            self.evaluate_acc_with_image(model, file_name, model_name, max_tokens)
        else:
            raise ValueError(f"Método de avaliação inválido: {method}")

    def convert_image_to_base64(self, image_path):
        """
        Converte uma imagem em base64.
        """
        try:
            with open(image_path, "rb") as image_file:
                image_base64 = base64.b64encode(image_file.read()).decode("utf-8")
            return image_base64
        except FileNotFoundError:
            print(f"Arquivo de imagem não encontrado: {image_path}")
            return ""

    def lc_access_with_image(self, model, ans, max_tokens):
        """
        Envia texto e opcionalmente uma imagem para a LLM.
        """
        return model.invoke(ans, max_tokens=max_tokens).content

    def lc_access(self, model, ans, max_tokens):
        """
        Envia texto e opcionalmente uma imagem para a LLM.
        """
        return model.invoke(ans, max_tokens=max_tokens).content

    def get_ans(self, ans, model, max_tokens=50):
        response = self.lc_access(model, ans, max_tokens)
        options_list = [
            (response.count("A"), "A"),
            (response.count("B"), "B"),
            (response.count("C"), "C"),
            (response.count("D"), "D"),
            (response.count("E"), "E"),
        ]
        sorted_options = sorted(options_list, reverse=True)
        return [option for _, option in sorted_options[:3]]

    def get_ans_by_accuracy(self, ans, model, max_tokens, images=[]):
        response = self.lc_access(model, ans, max_tokens, images)

        # Procurar correspondências com regex
        match_1 = re.findall(r"(?:|[Ll]etra |[Aa]lternativa )([ABCDE])\)", response)
        match_2 = re.findall(r"(?:|[Ll]etra |[Aa]lternativa )([ABCDE])", response)
        match_3 = re.findall(r"{([ABCDE])}", response)

        if len(match_3) > 0:
            response = match_3[-1]
        elif len(match_1) > 0:
            response = match_1[-1]
        elif len(match_2) > 0:
            response = match_2[-1]
        else:
            return ""  # Retorna vazio se nenhuma resposta válida for encontrada

        return response

    def evaluate_acc_with_image(self, model, file_name, model_name, max_tokens=50):
        """
        Avalia a acurácia do modelo com suporte a imagens.
        """
        init_date = datetime.datetime.now()

        print("Testando dados do dataset...")
        print(self.dataset)
        processor = DataProcessor()
        dataset = processor.read_data(path=self.dataset, type="jsonl")

        bar = tqdm(enumerate(dataset['train']), total=len(dataset['train']))

        correct_answers = 0
        total_questions = 0
        interactions = []

        for i, data in bar:
            question_text = data["text"]
            image_path = data.get(
                "image_path", None
            )  # Supondo que o dataset tenha imagens associadas
            expected_answer = data["answer"]

            llm_answer = self.get_ans_by_accuracy(
                question_text, model, max_tokens
            )

            if llm_answer == "":
                continue  # Pula para a próxima pergunta

            if llm_answer == expected_answer:
                correct_answers += 1

            total_questions += 1

            interactions.append(
                {
                    "question": question_text,
                    "image_path": image_path,
                    "expected_answer": expected_answer,
                    "llm_answer": llm_answer,
                }
            )

        accuracy = correct_answers / total_questions if total_questions > 0 else 0.0
        finish_date = datetime.datetime.now()

        os.makedirs(self.interaction_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        interaction_file = os.path.join(
            self.interaction_dir, file_name.replace(".csv", "_interactions.csv")
        )
        with open(interaction_file, mode="w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=["question", "image_path", "expected_answer", "llm_answer"],
            )
            writer.writeheader()
            writer.writerows(interactions)

        print(f"Interações gravadas em {interaction_file}")

        results_file = os.path.join(self.results_dir, file_name)
        try:
            df = pd.read_csv(results_file)
        except FileNotFoundError:
            df = pd.DataFrame(
                columns=[
                    "modelo",
                    "created_at",
                    "updated_at",
                    "init_date",
                    "finish_date",
                    "accuracy",
                ]
            )

        new_data = pd.DataFrame(
            [
                {
                    "modelo": model_name,
                    "created_at": init_date,
                    "updated_at": finish_date,
                    "init_date": init_date,
                    "finish_date": finish_date,
                    "accuracy": accuracy,
                }
            ]
        )

        if df.empty:
            df = new_data
        else:
            df = pd.concat([df, new_data], ignore_index=True)

        df.to_csv(results_file, index=False)
        print(f"Resultados gravados em {results_file}")

    def evaluate_acc(self, model, file_name, model_name, max_tokens=50):
        """
        Avalia a acurácia do modelo, limitando o número de tokens de saída e salvando os resultados.

        Args:
            model: Objeto do modelo LLM.
            file_name: Nome base do arquivo para salvar os resultados.
            model_name: Nome do modelo sendo avaliado.
            max_tokens: Número máximo de tokens permitidos na saída da LLM.
        """
        init_date = datetime.datetime.now()

        processor = DataProcessor()
        dataset = processor.read_data(path=self.dataset)
        bar = tqdm(enumerate(dataset["train"]), total=len(dataset["train"]))

        correct_answers = 0
        total_questions = 0

        interactions = []

        for i, data in bar:
            llm_answer = self.get_ans_by_accuracy(
                model, data["text"], model, max_tokens=max_tokens
            )
            expected_answer = data["answer"]

            if llm_answer == "":
                continue  # Pula para a próxima pergunta

            if llm_answer == expected_answer:
                correct_answers += 1

            total_questions += 1

            interactions.append(
                {
                    "question": data["text"],
                    "expected_answer": expected_answer,
                    "llm_answer": llm_answer,
                }
            )

        accuracy = correct_answers / total_questions if total_questions > 0 else 0.0

        finish_date = datetime.datetime.now()

        os.makedirs(self.interaction_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        interaction_file = os.path.join(
            self.interaction_dir, file_name.replace(".csv", "_interactions.csv")
        )
        with open(interaction_file, mode="w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(
                csv_file, fieldnames=["question", "expected_answer", "llm_answer"]
            )
            writer.writeheader()
            writer.writerows(interactions)

        print(f"Interações gravadas em {interaction_file}")

        results_file = os.path.join(self.results_dir, file_name)
        try:
            df = pd.read_csv(results_file)
        except FileNotFoundError:
            df = pd.DataFrame(
                columns=[
                    "modelo",
                    "created_at",
                    "updated_at",
                    "init_date",
                    "finish_date",
                    "accuracy",
                ]
            )

        new_data = pd.DataFrame(
            [
                {
                    "modelo": model_name,
                    "created_at": init_date,
                    "updated_at": finish_date,
                    "init_date": init_date,
                    "finish_date": finish_date,
                    "accuracy": accuracy,
                }
            ]
        )

        if df.empty:
            df = new_data
        else:
            df = pd.concat([df, new_data], ignore_index=True)

        df.to_csv(results_file, index=False)
        print(f"Resultados gravados em {results_file}")

    def evaluate_apk(self, model, file_name, model_name, max_tokens=5):
        """
        Avalia o modelo e salva as interações e os resultados em pastas separadas.

        Args:
            model: Objeto do modelo LLM.
            file_name: Nome base do arquivo para salvar os resultados.
            model_name: Nome do modelo sendo avaliado.
            max_tokens: Número máximo de tokens permitidos na saída da LLM.
        """
        init_date = datetime.datetime.now()

        processor = DataProcessor()
        dataset = processor.read_data(path=self.dataset)
        bar = tqdm(enumerate(dataset["train"]), total=len(dataset["train"]))

        aps = []
        interactions = []

        for i, data in bar:
            ans_list = self.get_ans(data["text"], model, max_tokens=max_tokens)
            expected_answer = data["answer"]

            average_precision = self.metrics.apk([expected_answer], ans_list, k=3)
            aps.append(average_precision)

            interactions.append(
                {
                    "question": data["text"],
                    "expected_answer": expected_answer,
                    "llm_answer_1": ans_list[0],
                    "llm_answer_2": ans_list[1] if len(ans_list) > 1 else "",
                    "llm_answer_3": ans_list[2] if len(ans_list) > 2 else "",
                }
            )

        mean_average_precision = np.mean(aps) if aps else 0.0

        finish_date = datetime.datetime.now()

        os.makedirs(self.interaction_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        interaction_file = os.path.join(
            self.interaction_dir, file_name.replace(".csv", "_interactions.csv")
        )
        with open(interaction_file, mode="w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=[
                    "question",
                    "expected_answer",
                    "llm_answer_1",
                    "llm_answer_2",
                    "llm_answer_3",
                ],
            )
            writer.writeheader()
            writer.writerows(interactions)

        print(f"Interações gravadas em {interaction_file}")

        results_file = os.path.join(self.results_dir, file_name)
        try:
            df = pd.read_csv(results_file)
        except FileNotFoundError:
            df = pd.DataFrame(
                columns=[
                    "modelo",
                    "created_at",
                    "updated_at",
                    "init_date",
                    "finish_date",
                    "mean_average_precision",
                ]
            )

        new_data = pd.DataFrame(
            [
                {
                    "modelo": model_name,
                    "created_at": init_date,
                    "updated_at": finish_date,
                    "init_date": init_date,
                    "finish_date": finish_date,
                    "mean_average_precision": mean_average_precision,
                }
            ]
        )

        if df.empty:
            df = new_data
        else:
            df = pd.concat([df, new_data], ignore_index=True)

        df.to_csv(results_file, index=False)
        print(f"Resultados gravados em {results_file}")

    def evaluate_with_metrics(self, model, file_name, model_name, max_tokens=50):
        """
        Avalia o modelo e calcula várias métricas, salvando os resultados e interações.

        Args:
            model: Objeto do modelo LLM.
            file_name: Nome base do arquivo para salvar os resultados.
            model_name: Nome do modelo sendo avaliado.
            max_tokens: Número máximo de tokens permitidos na saída da LLM.
        """
        init_date = datetime.datetime.now()

        processor = DataProcessor()
        dataset = processor.read_data(path=self.dataset)
        bar = tqdm(enumerate(dataset["train"]), total=len(dataset["train"]))

        true_answers = []
        predicted_answers = []

        interactions = []

        for i, data in bar:
            llm_answer = self.get_ans_by_accuracy(
                data["text"], model, max_tokens=max_tokens, images=data['images'].split(',')
            )
            expected_answer = data["answer"]

            if not llm_answer:
                print(
                    f"[Aviso] Nenhuma resposta válida encontrada para a pergunta: {data['text']}"
                )
                continue  # Pula para a próxima pergunta

            true_answers.append(expected_answer)
            predicted_answers.append(llm_answer)

            interactions.append(
                {
                    "question": data["text"],
                    "expected_answer": expected_answer,
                    "llm_answer": llm_answer,
                }
            )

        if len(true_answers) == 0 or len(predicted_answers) == 0:
            accuracy = precision = recall = f1 = 0.0
        else:
            accuracy = accuracy_score(true_answers, predicted_answers)
            precision = precision_score(
                true_answers, predicted_answers, average="weighted", zero_division=0
            )
            recall = recall_score(
                true_answers, predicted_answers, average="weighted", zero_division=0
            )
            f1 = f1_score(
                true_answers, predicted_answers, average="weighted", zero_division=0
            )

        finish_date = datetime.datetime.now()

        os.makedirs(self.interaction_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        interaction_file = os.path.join(
            self.interaction_dir, file_name.replace(".csv", "_interactions.csv")
        )
        with open(interaction_file, mode="w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(
                csv_file, fieldnames=["question", "expected_answer", "llm_answer"]
            )
            writer.writeheader()
            writer.writerows(interactions)

        print(f"Interações gravadas em {interaction_file}")

        results_file = os.path.join(self.results_dir, file_name)
        try:
            df = pd.read_csv(results_file)
        except FileNotFoundError:
            df = pd.DataFrame(
                columns=[
                    "modelo",
                    "created_at",
                    "updated_at",
                    "init_date",
                    "finish_date",
                    "accuracy",
                    "precision",
                    "recall",
                    "f1",
                ]
            )

        new_data = pd.DataFrame(
            [
                {
                    "modelo": model_name,
                    "created_at": init_date,
                    "updated_at": finish_date,
                    "init_date": init_date,
                    "finish_date": finish_date,
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            ]
        )

        if df.empty:
            df = new_data
        else:
            df = pd.concat([df, new_data], ignore_index=True)

        df.to_csv(results_file, index=False)
        print(f"Resultados gravados em {results_file}")
