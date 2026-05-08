from modalities.text.params import TextParams
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class TextGenerator:
    MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_ID,
            dtype=torch.bfloat8,
            device_map="auto"
        )

    def generate_from_topic(self, topic: str, params: TextParams) -> str:
        prompt = (
            f"Write 2-4 paragraphs about: {topic}\n\n"
            "Constraints:\n"
            f"- Sentence length: avg {params.sentence_length} words (6–24 range)\n"
            f"- Vocabulary level: grade {params.vocab_complexity}\n"
            f"- Tone: {params.emotional_tone} (neutral / warm / urgent / playful)\n"
            f"- Format: {params.structure} (prose / bullets / numbered)\n\n"
            "Rules:\n"
            "- Keep writing natural and coherent\n"
            "- Do not explain or mention these rules\n"
            "- Follow tone and structure strictly\n"
            "- Output ONLY the final text\n"
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

    def refine(self, text: str, params: TextParams, instruction: str) -> str:
        """
        TODO: Call the LLM with `text` and `instruction` (a plain-English
        description of what to improve, produced by the optimizer) to produce
        a refined version that preserves meaning but improves neural engagement.
        The system prompt should instruct the model to keep the same topic and
        approximate length while applying the changes described in `instruction`.
        Returns the refined text string.
        """
        # TODO: replace with real LLM call
        return f"{text}\n\n[MOCK REFINEMENT — instruction: {instruction}]"
