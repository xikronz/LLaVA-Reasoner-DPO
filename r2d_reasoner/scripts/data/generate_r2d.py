from datasets import load_dataset
from PIL import Image
from IPython.display import display
from groq import Groq
from openai import OpenAI
import base64
import json

class Generator:
    def __init__(self):
        self.gemma_key = ""
        self.oai_key = ""
        self.grok_key = ""
        self.image_root = ""
        self.tool = "oai"

    def get_client(self):
        match self.tool:
            case "oai":
                client = OpenAI(self.oai_key)
            case "groq":
                client = Groq(self.grok_key)
                return client
            case _:
                raise ValueError
            
    def encode_image(self, path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
        
    def generate_response(self, path):
        image_b64 = self.encode_image(self.image_root+path)
        client = self.get_client()

        response = client.responses.create(
            model="gpt-4o",
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{image_b64}"
                        },
                        {
                            "type": "input_text",
                            "text": "What is happening in this image?"
                        }
                    ]
                }
            ],
            max_output_tokens=2048
        )

        return (response.output_text)
    
    def get_img_path (self, sample):
        return sample[1]['content'][0]['image']
    
    def get_question (self, sample):
        return sample[1]['content'][1]['text']
    
def main():
    generator = Generator()

    with open("outputs/Qwen2.5-VL-3B-Instruct/Inference/r1_incorrect.json", "r") as f:
        data = json.load(f)
    
    print(data[0])


    
if __name__ == "__main__":
    main()

[{'role': 'system', 'content': [{'type': 'text', 'text': 'You are a Vision Language Model specialized in interpreting and analyzing visual information from image data. Given an image, and a question, provide a detailed explanation based on visual evidence present in the image. Reference specific, visible elements (e.g., signs, people, objects, colors, or positions) to support your reasoning and number your thoughts sequentially. Conclude with the final answer, clearly wrapped in the format: \n\n### Answer: {your answer here}'}]}, {'role': 'user', 'content': [{'type': 'image', 'image': 'data/images/sqa/6179.png'}, {'type': 'text', 'text': "Select the organism in the same genus as the Canada lynx.\nA. Lithobates catesbeianus\nB. Lynx pardinus\nC. Castor canadensis\nAnswer with the option's letter from the given choices directly."}]}, {'role': 'assistant', 'content': [{'type': 'text', 'text': 'B. Lynx pardinus'}]}, {'role': 'Qwen2.5-VL-3B-Sft', 'content': [{'type': 'text'}, {'text': 'The Canada lynx is a member of the family Felidae, which includes other species such as the bobcat, jaguar, and leopard. Among the options provided:\n\n- A. Lithobates catesbeianus is a species of frog.\n- B. Lynx pardinus is another species of lynx, which is in the same genus as the Canada lynx.\n- C. Castor canadensis is a species of beaver.\n\nSince the Canada lynx belongs to the genus Lynx, the organism in the same genus as the Canada lynx is Lynx pardinus.\n\n### Answer: B'}]}, {'role': 'Qwen2.5-Vl-3B-Instruct', 'content': [{'type': 'text'}, {'text': 'B'}]}]