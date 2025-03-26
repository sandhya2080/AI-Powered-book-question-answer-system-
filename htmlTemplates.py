import base64

with open("robo.png", "rb") as image_file:
    base64_user_image = base64.b64encode(image_file.read()).decode('utf-8')

with open("robo.png", "rb") as image_file:
    base64_bot_image = base64.b64encode(image_file.read()).decode('utf-8')

css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
'''

bot_template = f'''
<div class="chat-message bot">
    <div class="avatar">
        <img src="data:image/png;base64,{base64_bot_image}" style="max-height: 95px; max-width: 95px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{{{MSG}}}}</div>
</div>
'''

user_template = f'''
<div class="chat-message user">
    <div class="avatar">
        <img src="data:image/png;base64,{base64_user_image}">
    </div>    
    <div class="message">{{{{MSG}}}}</div>
</div>
'''

