from dotenv import load_dotenv
import os
import time
import argparse
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from langchain_anthropic import ChatAnthropic
from evaluete import Evaluete


def get_args():
    """
    Configura e retorna os argumentos da linha de comando.
    """
    parser = argparse.ArgumentParser(description="Avaliação de modelos de linguagem.")
    parser.add_argument(
        "--ds_path", type=str, required=True, help="Caminho para o dataset."
    )
    parser.add_argument(
        "--models",
        type=str,
        required=True,
        help="Modelos a serem avaliados, no formato 'nome_modelo:tipo_modelo'.",
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["accuracy", "metrics", "apk"],
        help="Método de avaliação: accuracy, metrics ou apk.",
    )
    return parser.parse_args()


def get_models(models_str):
    """
    Retorna uma lista de tuplas (nome_modelo, instância_do_modelo) com base na string de modelos fornecida.
    """
    models = []
    for model_info in models_str.split(","):
        name, model_type = model_info.split(":")
        if model_type == "groq":
            models.append((name, ChatGroq(model=name)))
        elif model_type == "openai":
            models.append((name, ChatOpenAI(model=name)))
        elif model_type == "mistral":
            models.append((name, ChatMistralAI(model=name)))
        elif model_type == "anthropic":
            models.append((name, ChatAnthropic(model=name)))
        else:
            raise ValueError(f"Tipo de modelo não suportado: {model_type}")
    return models


def evaluate_models(models, eval, args):
    """
    Avalia os modelos fornecidos.
    """
    for name, model in models:
        try:
            print(
                f"Executando avaliação para o modelo: {name} com método: {args.method}"
            )
            current_date = time.strftime("%Y-%m-%d-%H-%M-%S")
            file_name = f"{name}_{current_date}_{args.ds_path.split('/')[-1].split('.')[0]}_eval_{args.method}.csv"
            eval.evaluate(model, args.method, file_name, model_name=name, max_tokens=1)
            print("Modelo avaliado com sucesso!")
        except Exception as e:
            print(f"Erro ao avaliar o modelo: {name}")
            print(100 * "#")
            print(e)
            print(100 * "#")


if __name__ == "__main__":
    load_dotenv()

    # Obtém os argumentos da linha de comando
    args = get_args()
    print(args)

    # Inicializa o avaliador com o ds_path fornecido
    eval = Evaluete(args.ds_path)

    # Obtém a lista de modelos
    models = get_models(args.models)

    # Avalia cada modelo
    evaluate_models(models, eval, args)
