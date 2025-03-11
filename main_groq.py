from dotenv import load_dotenv
import os
import time
import argparse
from evaluete_client import EvalueteClient


def get_args():
    """
    Configura e retorna os argumentos da linha de comando.
    """
    parser = argparse.ArgumentParser(description="Avaliação de modelos de linguagem.")
    parser.add_argument(
        "--ds_path", type=str, required=True, help="Caminho para o dataset."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Modelos a serem avaliados, no formato 'nome_modelo'.",
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["accuracy", "metrics", "apk", "accuracy_with_image"],
        help="Método de avaliação: accuracy, metrics ou apk.",
    )
    parser.add_argument(
        "--max_tokens",
        type=str,
        required=True,
        help="Número máximo de tokens de retorno: 1, 2, 3, ...",
    )
    return parser.parse_args()


def evaluate_model(model, eval, args):
    """
    Avalia os modelos fornecidos.
    """
    try:
        print(
            f"Executando avaliação para o modelo: {model} com método: {args.method}"
        )
        current_date = time.strftime("%Y-%m-%d-%H-%M-%S")
        file_name = f"{model}_{current_date}_{args.ds_path}_eval_{args.method}.csv"
        eval.evaluate(
        method=args.method, 
        file_name=file_name, 
        model_name=model, 
        max_tokens=args.max_tokens
        )
        print("Modelo avaliado com sucesso!")
    except Exception as e:
        print(f"Erro ao avaliar o modelo")
        print(100 * "#")
        print(e)
        print(100 * "#")


if __name__ == "__main__":
    load_dotenv()

    # Obtém os argumentos da linha de comando
    args = get_args()

    # Inicializa o avaliador com o ds_path fornecido
    data_dir = os.getenv("DATA_PATH")
    eval = EvalueteClient(f"{data_dir}/{args.ds_path}")

    # Obtém a lista de modelos
    model = args.model

    # Avalia cada modelo
    evaluate_model(model, eval, args)
