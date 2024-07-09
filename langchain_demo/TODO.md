1. I need to get a new API because I accidentally pushed it to GitHub
2. Integrate splitting process with Sonia's DF process
3. See if you can make decision process more robust and not as rigid (look into agents)
4. Make a doc/tutorial for how gradio works
5. Change the chat history cell size to expand to the full length of the page 

---
1. Made the text cell longer. I am currently setting the height explicitly. I tried writing CSS code to make the hight dynamic but it didn't seem to work smoothly.
2. Modified the input cell to clear content after pressing enter. 
3. Combined splitting with Sonia's Data frame. Parser would not work until I used GPT-4. I tried a variety of error handlers but it did not seem to work. Also, for some reason it says that the data frame does not have pr at daily resolution but this is wrong. I try many methods to address this but couldn't figure a solution out yet.
4. Implemented agents to decide when to split and retrieve.