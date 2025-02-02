# Instruções de uso 

## 1. Instalação

Para instalar o projeto, siga os passos abaixo:

1. Clone o repositório:
```bash
git clone
```

2. Instale as dependências:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. Execute o projeto:
```bash
python main.py --ds_path ./data/train_1_of_100.csv --models "llama3-70b-8192:groq"
```