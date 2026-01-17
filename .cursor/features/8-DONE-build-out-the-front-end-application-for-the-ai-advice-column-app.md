# Build Out The Front End Application For The Ai Advice Column App

## Working directory

`~/Desktop/friendly-advice-columnist`

## Contents

This repo is an AI-powered advice columnist app. Right now, I have the back-end where the user can submit an interpersonal question and then get a response in my voice. The questions have to be related to interpersonal and relationship issues, such as marriage, divorce, families, kids, friends, etc. There are two LLMs that my app uses:
1. A base LLM
2. A fine-tuned LLM

When the user inputs a question, it goes to the base LLM which then generates a response. That response then goes to the fine-tuned LLM, which issues a numerical grade, 1 through 10, to rate the response. It also determines a few strengths and weaknesses of the base LLM's response. It then emits a more polished, revised response to the original question which is an improvement of the draft response. The user only sees the final revised response. He or she does not see the intermediate steps, including the draft response by the base LLM, the numerical grade, or the strengths and weaknesses of the draft response. 

I want to build out a front-end around this current back-end implementation. For the MVP, the front-end should be fairly simple. It should simply involve a text box where the user enters in their question and then clicks a button or presses Enter. The user then waits for the program to generate a response. The program will emit its response, which will appear in the application to the user.  

The MVP should just be a simple input/output application. But later versions could do things such as:
- Allow users to save their questions and responses
- Provide a history of previous questions and responses
- Allow users to rate the responses and provide feedback
- Implement a chatbot interface for users to interact with the AI advice columnist

We need to figure out several things: 
- How will users authenticate and authorize access to the application?
- How will the application handle errors and edge cases?
- What are the performance requirements for the application?
- How will end-to-end testing work when it comes to this front-end application? 
- What will the security requirements be, and how will we implement them? 
- How will we ensure that the front-end application is scalable and can handle a large number of users?

## Acceptance criteria
- A basic front-end application that hits my current back-end with a user question and then spits out a polished response. 
- This front-ended app should utilize best practices when it comes to building an application around an OpenAI model. It should follow the relevant guidelines of OpenAI and the AI industry as a whole. 
- I would like to build this front-end application in the same repo if possible. 
- The front-end application should have tests. 
- Before actually building out the front-end application, Claude needs to work with me to create a plan for how this application will be built. This will involve a series of steps that I need to sign off on before we actually build the application. Since this is an involved project, we will need to work together closely, and Claude shouldn't do too much without checking with me first. 
- The front-end application should be responsive and mobile-friendly. 
- The front-end application should be accessible and follow best practices for accessibility. 
- The front-end application should be optimized for performance and should load quickly. 
- The front-end application should be secure and follow best practices for security. 
- The front-end application should be scalable and should be able to handle a large number of users. 

Before working on this application, please present me with a series of proposed steps and get my permission before starting.

<!-- DONE -->
