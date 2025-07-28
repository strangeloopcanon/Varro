import random

# Read headlines from file
with open("data/headlines.txt", "r") as f:
    lines = f.readlines()

# Parse headlines
headlines = []
for i in range(0, len(lines), 3):
    if i+1 < len(lines):
        title = lines[i].strip()
        link = lines[i+1].strip()
        headlines.append((title, link))

# Meta-learning prompt template
META_LEARNING_PROMPT = """
Headline: "{headline}"

Your task is to predict the next major development related to this story.

Please structure your response as follows:
1. Goal Hypothesis (g): What is the primary objective or outcome I should predict?
2. Prior Knowledge (K): What relevant information or patterns can I draw from?
3. Reward Model (θ): What would constitute a "good" prediction? Define key criteria.
4. Exploration Plan (π_probe): What information would be most valuable to gather to improve my prediction?
5. Prediction: Based on the above, what is the most likely next development?

Think carefully and provide a detailed, well-reasoned response.
""".strip()

# Select a random headline and generate prompt
selected_headline = random.choice(headlines)
prompt = META_LEARNING_PROMPT.format(headline=selected_headline[0])

print("Selected Headline:")
print(selected_headline[0])
print("\nGenerated Prompt:")
print(prompt)