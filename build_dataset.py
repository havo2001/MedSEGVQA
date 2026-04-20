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


questions_bank = [
    'What can you see in the image?',
    'Describe the key finding in the image.',
    'Is there anything wrong with the image?',
    'Describe the abnormality in the image.',
    'Describe the findings in this image.',
    'Is there any abnormality in the image?',
    'What is present in this image, and where is it located?',
    'How many polyps are in the image, and what do they look like?',
    'Describe any polyps in the image',
]


def sample_intent():
    intents = ["presence", "attribute", "location", "description"]
    weights = [0.25, 0.40, 0.20, 0.15]
    return random.choices(intents, weights=weights, k=1)[0]


system_prompt = """
You are generating question-answer annotations for a medical image dataset.

Each input is a ground-truth description of an image.

The description may describe:
- a visible finding, OR
- a normal image / absence of findings

--------------------------------
Core principle:
--------------------------------
The description is the ONLY ground truth.

--------------------------------
Critical constraints:
--------------------------------
- Do NOT add, infer, or hallucinate any information.
- Do NOT invent findings, attributes, locations, or diagnoses.
- Every question MUST be answerable strictly from the description.
- The answer MUST be logically consistent with the description.

--------------------------------
Task:
--------------------------------
Generate exactly 1 open-ended QA pair.

--------------------------------
Allowed intents:
--------------------------------
- presence
- attribute
- location
- description

You will be given the required intent. You MUST follow it exactly.

--------------------------------
Intent behavior:
--------------------------------

presence:
- ask whether something described (or its absence) is true

attribute:
- ask about exactly ONE characteristic explicitly supported by the description
- do NOT introduce new attributes

location:
- ask about location ONLY if location is explicitly mentioned
- do NOT invent locations

description:
- ask for a short grounded description of what is stated

--------------------------------
Open-ended answer rules:
--------------------------------
- must be one short sentence
- must answer ONLY the question
- must reuse wording from the description
- must not add new information

--------------------------------
Quality requirements:
--------------------------------
- Question must be natural and unambiguous
- Question must NOT require external knowledge
- Question must NOT contradict the description
- Question must NOT assume a finding if none is stated

--------------------------------
Output format:
--------------------------------
Return valid JSON only:

{
  "qas": [
    {"type": "open_ended", "question": "...", "answer": "..."}
  ]
}
"""


USER_PROMPT_TEMPLATE = """
Ground-truth description:
"{prompt}"

Generate exactly 1 open-ended QA pair.

Required intent: {open_intent}

Important:
- You MUST follow the required intent exactly
- Use ONLY information from the description
- Do NOT invent findings, attributes, or locations
- Return JSON only
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
    if not isinstance(qas, list) or len(qas) != 1:
        return False, "Must return exactly 1 QA"

    if qas[0].get("type") != "open_ended":
        return False, "QA type must be open_ended"

    open_question = qas[0].get("question")
    open_answer = qas[0].get("answer")

    if not isinstance(open_question, str) or not open_question.strip():
        return False, "Missing open-ended question"

    if not isinstance(open_answer, str) or not open_answer.strip():
        return False, "Missing open-ended answer"

    return True, None


def generate_qas(prompt_text: str):
    open_intent = sample_intent()

    user_prompt = USER_PROMPT_TEMPLATE.format(
        prompt=prompt_text,
        open_intent=open_intent,
    )

    def api_call():
        return client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
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
                return qas, open_intent

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

            results.append({
                "image": image_path,
                "mask": mask_path,
                "type": "baseline",
                "question": random.choice(questions_bank),
                "answer": prompt_text,
            })

            qas, open_intent = generate_qas(prompt_text)

            for qa in qas:
                results.append({
                    "image": image_path,
                    "mask": mask_path,
                    "type": qa["type"],
                    "question": qa["question"],
                    "answer": qa["answer"],
                })

            if idx % 50 == 0:
                print(f"Processed {idx + 1}/{len(input_data)}")

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


DATASET_ROOT = "dataset"


def generate_questions_for_dataset(dataset_name):
    folder = os.path.join(DATASET_ROOT, dataset_name)

    generate_qa(
        os.path.join(folder, "train_prompt.json"),
        os.path.join(folder, "train.json"),
        os.path.join(folder, "train_failures.json"),
    )

    generate_qa(
        os.path.join(folder, "val_prompt.json"),
        os.path.join(folder, "val.json"),
        os.path.join(folder, "val_failures.json"),
    )

    generate_qa(
        os.path.join(folder, "test_prompt.json"),
        os.path.join(folder, "test.json"),
        os.path.join(folder, "test_failures.json"),
    )


if __name__ == "__main__":
    # generate_questions_for_dataset("UWaterloo")
    # generate_questions_for_dataset("BKAI")
    # generate_questions_for_dataset("BUID")
    # generate_questions_for_dataset("BUSI")
    # generate_questions_for_dataset("ClinicDB")
    # generate_questions_for_dataset("ETIS")
    # generate_questions_for_dataset("ISIC")
    # generate_questions_for_dataset("CVC300")
    # generate_questions_for_dataset("ColonDB")
    # generate_questions_for_dataset("Kvasir-SEG")
    # generate_questions_for_dataset("BUSBRA")
    # generate_questions_for_dataset("BRISC")
    pass