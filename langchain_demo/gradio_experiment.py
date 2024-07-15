import gradio as gr
import random
import time

def user(user_message, history):
    return "", history + [[user_message, None]]

def bot(history):
    bot_message = random.choice(["How are you?", "I love you", "I'm very hungry"])
    history[-1][1] = ""
    for character in bot_message:
        history[-1][1] += character
        time.sleep(0.05)
        yield history

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(elem_id="chatbox")
    msg = gr.Textbox()
    clear = gr.Button("Clear")

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)

    gr.HTML("""
        <script>
            function adjustChatboxHeight() {
                var chatbox = document.getElementById("chatbox");
                if (chatbox) {
                    chatbox.style.maxHeight = "500px";  // Set max height to prevent it from growing too large
                    chatbox.style.overflowY = "auto";   // Enable vertical scrolling
                    chatbox.style.height = "auto";  // Reset height to auto
                    chatbox.style.height = chatbox.scrollHeight + "px";  // Set height based on content
                }
            }

            // Adjust height when the page loads
            window.onload = adjustChatboxHeight;

            // Adjust height whenever chat content changes
            document.addEventListener('DOMContentLoaded', () => {
                const observer = new MutationObserver(adjustChatboxHeight);
                const chatbox = document.getElementById("chatbox");
                if (chatbox) {
                    observer.observe(chatbox, { childList: true, subtree: true });
                }
            });
        </script>
    """)

demo.queue()
demo.launch()



with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")

    def user(user_message, history):
        return "", history + [[user_message, None]]

    def bot(history):
        bot_message = random.choice(["How are you?", "I'm very hungry"])
        history[-1][1] = ""
        for character in bot_message:
            history[-1][1] += character
            time.sleep(0.05)
            yield history

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)
    
demo.queue()
demo.launch()
