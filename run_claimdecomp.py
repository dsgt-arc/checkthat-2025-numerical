import json
import os
import time

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

class ClaimDecomp:
    def __init__(self, data_path, prompt_path, output_path):
        self.raw_data_dir = data_path
        self.prompt_path = prompt_path
        self.output_path = output_path
        self.raw_data =  None
        self.claims = None
        self.prompt = None

        load_dotenv()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def prepare_data(self):
        self._load_data()
        self._get_claims_with_ids()
        self._load_config_prompt()

    def run_decomposition(self):
        few_shot_examples = self.prompt["few_shot_examples"]
        system_message = self.prompt["system_message"]

        output_data = []

        for item in tqdm(self.claims):
            claim_id = item.get("claim_id")
            original_claim = item.get("original_claim")

            # Build conversation history with system message, few-shot examples, and the test claim.
            messages = [system_message] + few_shot_examples + [
                {
                    "role": "user",
                    "content": f"Claim ID: {claim_id}\nOriginal Claim: {original_claim}"
                }
            ]

            try:
                response = self.client.chat.completions.create(
                    model=self.prompt["model"],
                    messages=messages,
                    temperature=self.prompt["temperature"],
                    max_tokens=self.prompt["max_tokens"],
                    frequency_penalty=self.prompt["frequency_penalty"],
                    presence_penalty=self.prompt["presence_penalty"],
                    n=1
                )
                assistant_message = response.choices[0].message.content.strip()

                try:
                    result_json = json.loads(assistant_message)
                except json.JSONDecodeError:
                    # Fallback in case the response isn't valid JSON.
                    result_json = {
                        "claim_id": claim_id,
                        "original_claim": original_claim,
                        "question0": assistant_message,
                        "question1": "",
                        "question2": ""
                    }

                item.update(result_json)
                output_data.append(item)
                # Pause to avoid hitting rate limits.
                time.sleep(1)

            except Exception as e:
                print(f"Error processing claim_id {claim_id}: {e}")
                item.update({
                    "question0": "",
                    "question1": "",
                    "question2": ""
                })
                output_data.append(item)

        # -----------------------------
        # SAVE OUTPUT DATA
        # -----------------------------
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"Processing complete. Check '{self.output_path}' for results.")


    def _load_data(self, **kwargs):
        with open(self.raw_data_dir, **kwargs) as f:
            self.raw_data = json.load(f)
 
    def _get_claims_with_ids(self):
        """Produces a list of tuples (claim_id, claim) """
        self.claims = [{"claim_id":i, 
                        "original_claim":claim['claim'], 
                        "country_of_origin":claim['country_of_origin'],
                        "label":claim['label'],
                        "lang":claim["lang"],
                        "taxonomy_label":claim["taxonomy_label"]} for i, claim in enumerate(self.raw_data)]

    def _load_config_prompt(self):
        with open(self.prompt_path) as f:
            self.prompt = json.load(f)

def main():
    data_path = "data/English/val_claims_quantemp.json" #this is the original dataset with all the metadata
    prompt_path = "numerical/prompts/claimdecomp_prompt.json"
    output_path = "data/decomposition/val_claims_decomposed.json"

    val_decomp = ClaimDecomp(data_path, prompt_path, output_path)
    val_decomp.prepare_data()
    val_decomp.run_decomposition()

if __name__ == "__main__":
    main()
