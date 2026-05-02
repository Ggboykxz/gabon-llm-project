import json
import os

def format_to_instruction(input_path, output_path):
    """Convertit les données brutes au format Alpaca/Instruction pour le fine-tuning."""
    formatted_data = []
    if not os.path.exists(input_path):
        print(f"Fichier source {input_path} non trouvé.")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    for item in raw_data:
        text = f"### Instruction:\n{item['instruction']}\n\n### Réponse:\n{item['output']}"
        formatted_data.append({"text": text})

    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in formatted_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    print(f"✅ Dataset formaté sauvegardé dans {output_path}")

if __name__ == '__main__':
    # Exemple d'usage si un fichier raw existe
    # format_to_instruction('data/raw_gabon.json', 'data/gabon_dataset.jsonl')
    pass
