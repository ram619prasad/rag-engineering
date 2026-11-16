from datasets import load_dataset

dataset = load_dataset("allenai/qasper", split="train")

print("Total papers:", len(dataset))

paper = dataset[0]

print("Available fields:", paper.keys())
print("Paper ID:", paper["id"])
print("Title:", paper["title"])

full_text = paper["full_text"]

print("Type:", type(full_text))
print("Fields:", full_text.keys())

print("Number of sections:", len(full_text["section_name"]))
print("First section:", full_text["section_name"][0])
print("First section paragraphs count:", len(full_text["paragraphs"][0]))
print("First section paragraphs:", full_text["paragraphs"][0])


qas = paper["qas"]

print("QA fields:", qas.keys())
print("Number of questions:", len(qas["question"]))
print("First question:", qas["question"][0])
print("First question ID:", qas["question_id"][0])
print("First answer annotations:", qas["answers"][0])


gold_evidence = qas["answers"][0]["answer"][0]["evidence"][0]

for section_idx, paragraphs in enumerate(full_text["paragraphs"]):
    for paragraph_idx, paragraph in enumerate(paragraphs):
        if paragraph.strip() == gold_evidence.strip():
            print("Evidence found!")
            print("Section:", full_text["section_name"][section_idx])
            print("Section index:", section_idx)
            print("Paragraph index:", paragraph_idx)