import datetime
import pandas as pd
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

  def lc_access(self, model, ans, max_tokens):
    return model.invoke(ans, max_tokens=max_tokens).content
  
  def get_ans(self, ans, model, max_tokens=50):
        response = self.lc_access(model, ans, max_tokens)
        options_list = [
            (response.count('A'), 'A'),
            (response.count('B'), 'B'),
            (response.count('C'), 'C'),
            (response.count('D'), 'D'),
            (response.count('E'), 'E')
        ]
        sorted_options = sorted(options_list, reverse=True)
        return [option for _, option in sorted_options[:3]]
  
  def get_ans_by_accuracy(self, ans, model, target, max_tokens):
    response = self.lc_access(model, ans, max_tokens)
    # TODO: Solicitar a LLM que destaque a resposta
    match_1 = re.findall(r'(?:|[Ll]etra |[Aa]lternativa )([ABCDE])\.', response)
    match_2 = re.findall(r'(?:|[Ll]etra |[Aa]lternativa )([ABCDE])', response)
    print("get_ans_by_accuracy")
    print(f"ans: {ans}")
    print(f"response: {response}")
    print(f"match_1: {match_1}")
    print(f"match_2: {match_2}")
    print(f"target: {target}")

  def evaluate_acc(self, model, file_name, model_name, max_tokens=5):
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
    bar = tqdm(enumerate(dataset['train']), total=len(dataset['train']))

    correct_answers = 0
    total_questions = 0

    # Lista para armazenar as interações
    interactions = []

    for i, data in bar:
        # Obter a resposta da LLM com limite de tokens
        llm_answer = self.get_ans(data['text'], model, max_tokens=max_tokens)[0]  # Pega apenas a primeira resposta
        expected_answer = data['answer']

        # Verifica se a resposta da LLM é igual à resposta esperada
        if llm_answer == expected_answer:
            correct_answers += 1

        # Atualiza o total de perguntas processadas
        total_questions += 1

        # Salva as interações
        interactions.append({
            "question": data['text'],
            "expected_answer": expected_answer,
            "llm_answer": llm_answer
        })

    # Calcula a acurácia
    accuracy = correct_answers / total_questions if total_questions > 0 else 0.0

    finish_date = datetime.datetime.now()

    # Define os caminhos para salvar os arquivos
    interaction_dir = "interaction"
    results_dir = "results"

    # Cria as pastas se elas não existirem
    os.makedirs(interaction_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Arquivo para interações
    interaction_file = os.path.join(interaction_dir, file_name.replace(".csv", "_interactions.csv"))
    with open(interaction_file, mode="w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["question", "expected_answer", "llm_answer"])
        writer.writeheader()
        writer.writerows(interactions)

    print(f"Interações gravadas em {interaction_file}")

    # Arquivo para resultados
    results_file = os.path.join(results_dir, file_name)
    try:
        df = pd.read_csv(results_file)
    except FileNotFoundError:
        df = pd.DataFrame(columns=["modelo", "created_at", "updated_at", "init_date", "finish_date", "accuracy"])

    new_data = pd.DataFrame([{
        "modelo": model_name,
        "created_at": init_date,
        "updated_at": finish_date,
        "init_date": init_date,
        "finish_date": finish_date,
        "accuracy": accuracy
    }])

    # Verifica se o DataFrame está vazio antes de concatenar
    if df.empty:
        df = new_data
    else:
        df = pd.concat([df, new_data], ignore_index=True)

    df.to_csv(results_file, index=False)
    print(f"Resultados gravados em {results_file}")

  def evaluate(self, model, file_name, model_name, max_tokens=5):
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
    bar = tqdm(enumerate(dataset['train']), total=len(dataset['train']))

    aps = []
    interactions = []

    for i, data in bar:
        # Obter a resposta da LLM com limite de tokens
        ans_list = self.get_ans(data['text'], model, max_tokens=max_tokens)
        expected_answer = data['answer']

        # Calcula a precisão média para as respostas geradas
        average_precision = self.metrics.apk([expected_answer], ans_list, k=3)
        aps.append(average_precision)

        # Salva a interação
        interactions.append({
            "question": data['text'],
            "expected_answer": expected_answer,
            "llm_answer_1": ans_list[0],
            "llm_answer_2": ans_list[1] if len(ans_list) > 1 else "",
            "llm_answer_3": ans_list[2] if len(ans_list) > 2 else ""
        })

    # Calcula a precisão média geral
    mean_average_precision = np.mean(aps) if aps else 0.0

    finish_date = datetime.datetime.now()

    # Define os caminhos para salvar os arquivos
    interaction_dir = "interaction"
    results_dir = "results"

    # Cria as pastas se elas não existirem
    os.makedirs(interaction_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Arquivo para interações
    interaction_file = os.path.join(interaction_dir, file_name.replace(".csv", "_interactions.csv"))
    with open(interaction_file, mode="w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["question", "expected_answer", "llm_answer_1", "llm_answer_2", "llm_answer_3"])
        writer.writeheader()
        writer.writerows(interactions)

    print(f"Interações gravadas em {interaction_file}")

    # Arquivo para resultados
    results_file = os.path.join(results_dir, file_name)
    try:
        df = pd.read_csv(results_file)
    except FileNotFoundError:
        df = pd.DataFrame(columns=["modelo", "created_at", "updated_at", "init_date", "finish_date", "mean_average_precision"])

    new_data = pd.DataFrame([{
        "modelo": model_name,
        "created_at": init_date,
        "updated_at": finish_date,
        "init_date": init_date,
        "finish_date": finish_date,
        "mean_average_precision": mean_average_precision
    }])

    # Verifica se o DataFrame está vazio antes de concatenar
    if df.empty:
        df = new_data
    else:
        df = pd.concat([df, new_data], ignore_index=True)

    df.to_csv(results_file, index=False)
    print(f"Resultados gravados em {results_file}")

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

  def evaluate_with_metrics(self, model, file_name, model_name, max_tokens=5):
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
      bar = tqdm(enumerate(dataset['train']), total=len(dataset['train']))

      # Armazena respostas verdadeiras e previstas
      true_answers = []
      predicted_answers = []

      # Lista para armazenar as interações
      interactions = []

      for i, data in bar:
          # Obter a resposta da LLM com limite de tokens
          ans_list = self.get_ans(data['text'], model, max_tokens=max_tokens)
          predicted_answer = ans_list[0]  # Usa a primeira opção como principal previsão
          true_answer = data['answer']

          # Armazena as respostas para cálculo de métricas
          true_answers.append(true_answer)
          predicted_answers.append(predicted_answer)

          # Salva a interação
          interactions.append({
              "question": data['text'],
              "expected_answer": true_answer,
              "llm_answer_1": ans_list[0],
              "llm_answer_2": ans_list[1] if len(ans_list) > 1 else "",
              "llm_answer_3": ans_list[2] if len(ans_list) > 2 else ""
          })

      # Cálculo das métricas
      accuracy = accuracy_score(true_answers, predicted_answers)
      precision = precision_score(true_answers, predicted_answers, average='weighted', zero_division=0)
      recall = recall_score(true_answers, predicted_answers, average='weighted', zero_division=0)
      f1 = f1_score(true_answers, predicted_answers, average='weighted', zero_division=0)

      finish_date = datetime.datetime.now()

      # Define os caminhos para salvar os arquivos
      interaction_dir = "interaction"
      results_dir = "results"

      # Cria as pastas se elas não existirem
      os.makedirs(interaction_dir, exist_ok=True)
      os.makedirs(results_dir, exist_ok=True)

      # Arquivo para interações
      interaction_file = os.path.join(interaction_dir, file_name.replace(".csv", "_interactions.csv"))
      with open(interaction_file, mode="w", encoding="utf-8", newline="") as csv_file:
          writer = csv.DictWriter(csv_file, fieldnames=["question", "expected_answer", "llm_answer_1", "llm_answer_2", "llm_answer_3"])
          writer.writeheader()
          writer.writerows(interactions)

      print(f"Interações gravadas em {interaction_file}")

      # Arquivo para resultados
      results_file = os.path.join(results_dir, file_name)
      try:
          df = pd.read_csv(results_file)
      except FileNotFoundError:
          df = pd.DataFrame(columns=["modelo", "created_at", "updated_at", "init_date", "finish_date", "accuracy", "precision", "recall", "f1"])

      new_data = pd.DataFrame([{
          "modelo": model_name,
          "created_at": init_date,
          "updated_at": finish_date,
          "init_date": init_date,
          "finish_date": finish_date,
          "accuracy": accuracy,
          "precision": precision,
          "recall": recall,
          "f1": f1
      }])

      # Verifica se o DataFrame está vazio antes de concatenar
      if df.empty:
          df = new_data
      else:
          df = pd.concat([df, new_data], ignore_index=True)

      df.to_csv(results_file, index=False)
      print(f"Resultados gravados em {results_file}")
