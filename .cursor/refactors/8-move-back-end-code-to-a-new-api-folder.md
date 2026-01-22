---
github_issue: 2
---
# Move Back End Code To A New Api Folder

## Working directory

`~/Desktop/friendly-advice-columnist`

## Contents

We have our front-end code for this repo in an app/ directory. It would be good to have a parallel structure with our back-end code. This includes the code that handles the back-end and API logic. This code should be in a new directory called api/. There should also be a claude.md at the root of this directory outlining the file structure, technology used, and other important points, just like there is today at the root of the app directory. 

We might also want to consider a third directory, ai/. This directory will include all of the training-related material and also all of the model code. We would have to decide where to put files like run_inference.py Since they really overlap both with AI and APIs.  

## Acceptance criteria

- A new folder called api/ With our backend and API logic. 
- A claude.md file describing the file structure and technologies used etc in the root of the api/ folder.
- Possibly another directory called ai/ For the AI model And trainings related logic.
- If we do create this directory, it should also have a cloud.md file with similar information as the others.   
