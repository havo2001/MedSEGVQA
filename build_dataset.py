import json
import time
import os
import random
import dotenv
import tqdm
from openai import OpenAI

dotenv.load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

random.seed(42)

def sample_intents():
    intents = ["presence", "attribute", "location", "description"]
    weights = [0.25, 0.40, 0.20, 0.15]

    yesno_intent = random.choices(intents, weights=weights, k=1)[0]
    open_intent = random.choices(intents, weights=weights, k=1)[0]

    return yesno_intent, open_intent


system_prompt = """
You are generating question-answer annotations for a medical image dataset.

Each input is a ground-truth description of a visible finding in a medical image.

Your task:
Generate exactly 2 question-answer pairs:
1. one yes/no question
2. one open-ended question

--------------------------------
Medical accuracy requirements:
--------------------------------
- The description is the ONLY ground truth.
- Do NOT add or infer any extra medical information.

--------------------------------
Grounding rules:
--------------------------------
- Only use attributes explicitly present in the description.
- Do NOT modify or generalize the description.

--------------------------------
Question generation rules:
--------------------------------
You will be given:
- required yes/no intent
- required open-ended intent

You MUST follow them exactly.

Allowed intents:
- presence
- attribute
- location
- description

--------------------------------
Intent behavior:
--------------------------------

presence:
- ask if the finding exists

attribute:
- ask about ONE characteristic (color, shape, size, number, etc.)

location:
- ask where it is located

description:
- ask for a short grounded description

--------------------------------
Answer rules:
--------------------------------
- Yes/No answer MUST be exactly: "Yes."
- Open-ended answer must:
  - be one short sentence
  - answer ONLY what is asked
  - reuse words from the description

--------------------------------
Output rules:
--------------------------------
- Return valid JSON only
- No explanation
- Return exactly:
{
  "qas": [
    {"type": "yes_no", "question": "...", "answer": "Yes."},
    {"type": "open_ended", "question": "...", "answer": "..."}
  ]
}
"""



USER_PROMPT_TEMPLATE = """
Ground-truth description:
"{prompt}"

Generate exactly 2 QA pairs:
- one yes/no
- one open-ended

Required yes/no intent: {yesno_intent}
Required open-ended intent: {open_intent}

Intent definitions:

- presence:
  ask about whether the finding exists

- attribute:
  ask about one characteristic (color, shape, size, number, etc.)

- location:
  ask where the finding is located

- description:
  ask for a short description of the finding

Important:
- You MUST follow the required intent exactly.
- Do NOT switch intent.
- Use only information from the description.
- Return JSON only.
"""

def call_with_retry(fn, max_retries=5, base_sleep=1.0):
    last_error = None
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            last_error = e
            if attempt == max_retries - 1:
                raise
            time.sleep(base_sleep * (2 ** attempt))
    raise last_error


def parse_qas(content):
    try:
        obj = json.loads(content)
    except Exception:
        return None

    return obj.get("qas")


def validate_qa(qas):
    if not isinstance(qas, list) or len(qas) != 2:
        return False, "Must return exactly 2 QA"

    if qas[0].get("type") != "yes_no" or qas[1].get("type") != "open_ended":
        return False, "Wrong order/types"

    if qas[0]["answer"] != "Yes.":
        return False, "Yes/No must be 'Yes.'"

    return True, None


def generate_qas(prompt_text: str):
    yesno_intent, open_intent = sample_intents()

    user_prompt = USER_PROMPT_TEMPLATE.format(
        prompt=prompt_text,
        yesno_intent=yesno_intent,
        open_intent=open_intent,
    )

    def api_call():
        return client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.4,
            max_tokens=300,
            response_format={"type": "json_object"},
        )

    for attempt in range(5):
        try:
            response = call_with_retry(api_call)
            content = response.choices[0].message.content

            qas = parse_qas(content)
            if qas is None:
                continue

            valid, _ = validate_qa(qas)
            if valid:
                return qas, yesno_intent, open_intent

        except Exception:
            pass

        time.sleep(1 + attempt)

    raise RuntimeError(f"Failed for prompt: {prompt_text}")


def generate_qa(json_file, output_file, failed_file=None):
    with open(json_file, "r") as f:
        input_data = json.load(f)

    results = []
    failures = []

    for idx, item in enumerate(tqdm.tqdm(input_data)):
        try:
            image_path = item["image"]
            mask_path = item["mask"]
            prompt_text = item["prompt"]

            # baseline
            results.append({
                "image": image_path,
                "mask": mask_path,
                "type": "baseline",
                "question": prompt_text,
                "answer": "It is segmented."
            })

            qas, yesno_intent, open_intent = generate_qas(prompt_text)

            for qa in qas:
                results.append({
                    "image": image_path,
                    "mask": mask_path,
                    "type": qa["type"],
                    "question": qa["question"],
                    "answer": qa["answer"]
                })

            if idx % 50 == 0:
                print(f"Processed {idx+1}/{len(input_data)}")

        except Exception as e:
            failures.append({
                "image": item.get("image"),
                "mask": item.get("mask"),
                "prompt": item.get("prompt"),
                "error": repr(e),
            })
            print(f"Failed: {e}")

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    if failed_file:
        with open(failed_file, "w") as f:
            json.dump(failures, f, indent=2)


def generate_questions_for_dataset(dataset_folder):
    train_json = os.path.join(dataset_folder, "train_prompt.json")
    val_json = os.path.join(dataset_folder, "val_prompt.json")
    test_json = os.path.join(dataset_folder, "test_prompt.json")

    generate_qa(
        train_json,
        os.path.join(dataset_folder, "train.json"),
        os.path.join(dataset_folder, "train_failures.json"),
    )

    generate_qa(    
        val_json,
        os.path.join(dataset_folder, "val.json"),
        os.path.join(dataset_folder, "val_failures.json"),
    )

    generate_qa(
        test_json,
        os.path.join(dataset_folder, "test.json"),
        os.path.join(dataset_folder, "test_failures.json"),
    )


if __name__ == "__main__":
    # generate_questions_for_dataset("UWaterloo")
    # generate_questions_for_dataset("BKAI")
    # generate_questions_for_dataset("BUID")
    # generate_questions_for_dataset("BUSI") - problem with the no tumor case
    