css = '''
<style>
.chat-message {
    display: grid;
    gap: 8px;
    grid-template-columns: 32px 1fr;
    align-item: center;
}

.chat-message.user {
    display: grid;
    gap: 8px;
    grid-template-columns: 0.8fr 32px;
    align-item: center;
    justify-content: flex-end;
}

.chat-message .message {
    border-radius: 10px;
    padding: 8px;
}

.chat-message.bot .message {
    background: #E5E0E080;
    border: 0.5px solid #49454F80;
    color: black;

}
.chat-message.user .message {
    border: 0.5px solid #49454F80;
    background: white;
    color: #49454F;
    justify-content: flex-end;
}

.chat-message .avatar {
  width: 32px;
  height: 32px;
  border-radius: 20px;
  overflow: hidden;
}

.chat-message .avatar img {
  display: block;
  width: 100%;
  height: 100%;
}

 .stApp {
        background: #F2FBF6;
    }

    header[data-testid="stHeader"] {
        display: none;
    }
    
    [data-testid="stSidebar"]::before {
        content: "";
        display: block;
        width: 240px;
        height: 80px;
        background-image: url('app/static/logo.jpg');
        margin: 35px auto 0px;
    }

    [data-testid="stSidebar"] {
        top: 25px;
        left: 25px;
        height: calc(100% - 60px) !important; 
        border-radius: 10px;
        background: white;
    }

    [data-testid="stSidebar"] + section > div:first-child{
        position: absolute !important;
        top: 25px;
        left: 386px;
        right: 35px;
        height: calc(100% - 60px) !important; 
        background: white;
        max-width:initial;
        width: initial;
        border-radius: 10px;
        padding: 3rem 1rem 6rem;
    }

    [data-testid="stSidebar"] + section [data-testid="stVerticalBlock"] {
        position: static;
        max-height: 75vh;
        overflow-y: scroll;
        overflow-x: visible;
        width: max-content;
    }

    [data-testid="stSidebar"] + section [data-testid="stVerticalBlock"] > div  [data-testid="stMarkdownContainer"]{
        margin-bottom: 0px;
    }

    [data-testid="stSidebar"] + section [data-testid="stVerticalBlock"] > div:nth-child(2) [data-testid="stMarkdownContainer"] h2  {
        padding: 0px;
    }

    [data-testid="stSidebar"] + section [data-testid="stVerticalBlock"] > div:nth-child(2) [data-testid="stMarkdownContainer"] h2 span {
        color: black !important;
        font-size: 16px;
        font-weight: 400;
        line-height: 25px;
        background: #E5E0E080;
        border: 0.5px solid #49454F80;
        padding: 8px;
        border-radius: 8px;
    }

    [data-testid="stSidebar"] + section [data-testid="stVerticalBlock"] > div:nth-child(3) {
        position: absolute;
        bottom: 25px;
    }

    [data-testid="stSidebar"] + section [data-testid="stVerticalBlock"] > div:nth-child(3)  [data-baseweb="input"] {
        border: 1px solid #49454F80;
        border-radius: 4px;
        background: white;
        position: relative;
    }

    [data-testid="stSidebar"] + section [data-testid="stVerticalBlock"] > div:nth-child(3)  [data-baseweb="input"]::after {
        content: "";
        background-image: url("app/static/send.jpg");
        color: black !important;
        position: absolute;
        width: 24px;
        height: 24px;
        right: 12px;
        top: 50%;
        transform: translateY(-50%);
    }

    [data-testid="stSidebar"] + section [data-testid="stVerticalBlock"] > div:nth-child(3)  [data-baseweb="input"] input {
        background: white;
        padding: 16px;
        color: black;
    }

    [data-testid="stSidebar"] * {
        color: #8A8D90;
    }

    [data-testid="stSidebar"] > div > div:first-child {
        display: none;
    }

    [data-testid="stSidebar"] > div > div:nth-child(2) * {
        position: static !important;
    }

    section[data-testid="stFileUploadDropzone"] {
        background: white;
    }

    section[data-testid="stFileUploadDropzone"] + div {
        margin-block-start: -90px;
    }

    section[data-testid="stFileUploadDropzone"] * {
        background: white;
        color: white !important;
    }

    section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div:nth-child(3) button{
        position: absolute !important;
        bottom: 130px;
        left: 50%;
        transform: translateX(-50%);
        width: 106px;
        background: white;
        color: #06983E;
    }

    section[data-testid="stFileUploadDropzone"] button[kind="secondary"] {
        position: absolute !important;
        bottom: 150px;
        left: 50%;
        transform: translateX(-50%);
    }

    section[data-testid="stFileUploadDropzone"] button[kind="secondary"] {
        position: absolute !important;
        bottom: 180px;
        left: 50%;
        transform: translateX(-50%);
        background: linear-gradient(270deg, #0E7034 0%, #34AD6C 100%);
        color: white !important;
    }

    </style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="app/static/ai_avatar.jpg">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="message">{{MSG}}</div>
    <div class="avatar">
        <img src="app/static/human_avatar.jpg">
    </div>    
</div>
'''
