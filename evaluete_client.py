import base64
import datetime
from groq import Groq
import pandas as pd
from dotenv import load_dotenv
import csv
import re
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from data_processing import DataProcessor
from metrics import Metrics

class EvalueteClient:
    def __init__(self, dataset):
        self.metrics = Metrics()
        self.dataset = dataset
        load_dotenv()
        self.results_dir = os.getenv("RESULTS_PATH")
        self.interaction_dir = os.getenv("INTERACTION_PATH")

    def evaluate(self, method, file_name, model_name, max_tokens=5):
        if method == "metrics":
            self.evaluate_with_metrics(file_name, model_name, max_tokens)
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
        
    def make_message(self, text, images=[]):
        """
        Cria uma mensagem para ser enviada para a LLM.
        """
        message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": text,
                },
            ],
        }

        if images:
            image_path = f"data/{images[0]}"
            base64_image = self.convert_image_to_base64(image_path)
            message["content"].append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    },
                }
            )

        return message
    
    def make_message_url(self, text, images=[]):
        """
        Cria uma mensagem para ser enviada para a LLM.
        """
        message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": text,
                },
            ],
        }

        if images:
            image_path = images[0]
            message["content"].append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"{image_path}",
                    },
                }
            )

        return message

    def lc_access_with_image(self, llm, model_name, ans, max_tokens, images=[]):
        """
        Envia texto e opcionalmente uma imagem para a LLM.
        """
        chat_completion = llm.chat.completions.create(
            messages=[
                self.make_message_url(ans, images),
            ],
            model=model_name,
            max_tokens=int(max_tokens)
        )

        return chat_completion.choices[0].message.content
    
    def get_ans_by_accuracy(self, client, model, ans, max_tokens, images=[]):
        response = self.lc_access_with_image(client, model, ans, max_tokens, images)

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

    def evaluate_with_metrics(self, file_name, model_name, max_tokens=50):
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

        llm = Groq()

        for _, data in bar:
            images = data.get("images", "")
            if images:
                images = images.split(",")
            else:
                images = []

            llm_answer = self.get_ans_by_accuracy(
                client=llm,
                model=model_name,
                ans=data["text"],
                max_tokens=max_tokens, 
                images=images
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
