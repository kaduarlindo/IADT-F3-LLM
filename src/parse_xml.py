import os
import xml.etree.ElementTree as ET

def load_xml_data(xml_folder):
    data = []
    for filename in os.listdir(xml_folder):
        if filename.endswith(".xml"):
            path = os.path.join(xml_folder, filename)
            tree = ET.parse(path)
            root = tree.getroot()

            for item in root.findall(".//record"):
                symptoms = item.findtext("symptoms")
                treatment = item.findtext("treatment")
                if symptoms and treatment:
                    data.append({
                        "context": symptoms,
                        "question": "Qual é o tratamento indicado?",
                        "answer": treatment
                    })
    return data