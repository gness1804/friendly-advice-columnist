# Add Ability To Save Sessions And Record A History Of Questions That You Asked

## Working directory

`~/Desktop/friendly-advice-columnist`

## Contents

It would be nice to have a list of questions that the user previously asked in a menu, like  what ChatGPT does. For instance, if a user asks the friendly advice columnist three questions, each of those questions would be recorded on a side menu, and they could click into each of those questions to go back to the conversation. This would need to involve the ability to save sessions. Since I'm not implementing authentication just yet, it seems like that the session saving would need a solution such as local storage. This feature could persist past conversation history across sessions. There should also be a button to erase conversation history.  

## Acceptance criteria

- Users have the ability to view past conversations. 
- Past conversations are persisted across sessions as well as within a session. 
- Past conversations are displayed via selecting a menu option for each conversation, similar to what ChatGPT does. 
- A button exists that the user can press, which erases their conversation history. This button should require confirmation from the user to complete this action. 
