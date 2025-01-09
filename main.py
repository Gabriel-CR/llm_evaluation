from dotenv import load_dotenv
import os
import time
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from langchain_anthropic import ChatAnthropic
from evaluete import Evaluete

if __name__ == '__main__':
  load_dotenv()
  eval = Evaluete("./data/train_10.csv")

  models = [
    ("mixtral-8x7b-32768", ChatGroq(model="mixtral-8x7b-32768")),

  ]

  for name, model in models:
    try:
      print(f"Executando avaliação para o modelo: {name}")
      current_date = time.strftime("%Y-%m-%d-%H-%M-%S")
      # eval.evaluate_acc(model, name + current_date + ".csv", max_tokens=1, model_name=name)
      eval.evaluate(model, name + current_date + ".csv", max_tokens=1, model_name=name)
      print("Modelo avaliado com sucesso!")
    except Exception as e:
      print(f"Erro ao avaliar o modelo: {name}")
      print(100*"#")
      print(e)
      print(100*"#")
