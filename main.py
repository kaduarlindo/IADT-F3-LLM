from src.parse_xml import load_xml_data
from src.prepare_dataset import create_hf_dataset
from src.train_model import train_model
from src.api import app

XML_PATH = "./data"
MODEL_NAME = "pierreguillou/bert-large-cased-squad-v1.1-portuguese"

if __name__ == "__main__":
    print("🔹 Lendo XMLs...")
    data = load_xml_data(XML_PATH)
    dataset = create_hf_dataset(data)

    print("🔹 Iniciando treinamento...")
    train_model(dataset, MODEL_NAME)

    print("🔹 Subindo API em http://localhost:5000 ...")
    app.run(host="0.0.0.0", port=5000)