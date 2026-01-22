---
github_issue: 3
---
# Build An Ai Scorer That Benchmarks Model Outputs Against Each Other

## Working directory

`~/Desktop/build_llm_karpathy`

## Contents

I want to build an automated scorer that allows a user to run benchmarks of different AI models against each other. This scorer will take in arguments for Model A and Model B and will run a certain number of tests, default 3 for each model. By test, I mean it will run the MVP, as noted in @advice_mvp.py, n number of times against each model. It will then use a third LLM, Model C, to evaluate each of those outputs and score them. This will allow me to compare the outputs of different models against each other. For instance, if I want to score my newest fine-tuned model against the base model.

## Acceptance criteria
- a script that takes in arguments for each model to compare and runs the MVP script against each of them a certain number of times.
- After producing all of these outputs, a third model will score them and then tally up the average scores for each model.
- Users should be able to specify whether an output file is used and if so, where that file is saved to.
