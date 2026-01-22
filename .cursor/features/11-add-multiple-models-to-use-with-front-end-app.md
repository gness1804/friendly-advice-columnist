---
github_issue: 5
---
# Add Multiple Models To Use With Front End App

## Working directory

`~/Desktop/friendly-advice-columnist`

## Contents

Right now, the front-end application and the MVP as a whole only uses OpenAI models, the GPT-4.1 mini base model, and a fine-tuned model based on that base model. I want to add the ability to use other models. This would involve a drop-down in the front-end application to select a different model. On the back-end, this will require fine-tuning new models from other companies, such as Anthropic or Meta. Right now, I would want to keep the base model as GPT-4.1 mini, but the fine-tuned model would be selectable from several options. 

## Acceptance criteria

- The front-end application offers the ability to choose from multiple models using a dropdown. 
- The back-end will be able to point to different fine-tune models, not just one, based on what the user selects. 
- The CLI version of the application will also offer an option to use a different model such as Anthropic or Meta Llama.
- The fine-tuned model will be the one that's selectable. The base model will remain GPT-4.1 mini.  
