# Add Check To Ensure That Question Relates To Interpersonal And Relationship Issues

## Working directory

`~/Desktop/friendly-advice-columnist`

## Contents

This is an application that takes in interpersonal relationship questions and emits answers. But I don't believe there's currently a way to enforce that the questions have to be interpersonal. We need to add a layer right after the user submits their initial question to ensure that the question is actually relevant. This screener, which might be a different LLM, the base LLM, or something else, should read and evaluate the question. If the question is not related to interpersonal and relationship matters, it should error out and show an error to the user and not process the question. 

We should implement this because we don't want the app to try attempting questions about car repair, for instance. We want the applications only competency to be advice columns, and we don't want to answer any questions outside of its competency. 

## Acceptance criteria
- When the application receives a question that is not related to interpersonal and relationship matters, it does not process the question. Instead, the application errors out and shows an error message to the user asking them to please submit a question that's relevant to interpersonal and relationship matters.

<!-- DONE -->
