from modalities.text.params import TextParams
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class TextGenerator:
    MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_ID,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa"
        )
        self.model = torch.compile(self.model, backend="inductor")

    def generate_from_topic(self, topic: str, params: TextParams) -> str:
        prompt = (
            f"Write an engaging educational podcast script about: {topic}\n\n"
            "Requirements:\n"
            f"- Total length: approximately 700 words\n"
            f"- Use 4 well-developed paragraphs\n"
            f"- Each paragraph should be moderately detailed (roughly 120-220 words)\n"
            f"- Average sentence length: {params.sentence_length} words\n"
            f"- Tone: {params.emotional_tone}\n"
            f"- Complexity level: {params.vocab_complexity}\n"
            "- Focus on clarity, flow, and listener engagement\n"
            "- Include interesting explanations, examples, or analogies where appropriate\n"
            "- Avoid overly short paragraphs or fragmented ideas\n"
            "- Avoid excessive repetition or unnecessary filler\n"
            "- Write in a natural spoken style suitable for narration\n"
            "- Do not include section titles, bullet points, stage directions, or speaker labels\n"
            "- Output only the podcast narration text\n"
        )
        messages = [
            {
                "role": "system",
                "content": "You are a writing engine. Output ONLY the final text. Do not show reasoning, drafts, checks, or explanations."
            },
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=1000
            )
        
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

        content = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return content

    def refine(self, text: str, params: TextParams, instruction: str, system_prompt: str = "") -> str:
        user_content = f"{instruction}\n\nFull original text:\n\n{text}"
        messages = [
            {
                "role": "system",
                "content": system_prompt or "You are a precise text editor. Output ONLY the final text.",
            },
            {"role": "user", "content": user_content},
        ]
        text_input = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        model_inputs = self.tokenizer([text_input], return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=1500,
            )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        return self.tokenizer.decode(output_ids, skip_special_tokens=True)
