import uvicorn
from fastapi import FastAPI
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory, ConfigurableFieldSpec
from langchain_openai import ChatOpenAI
from langchain.schema import StrOutputParser
from langchain.prompts.chat import  ChatPromptTemplate
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.chat_message_histories import ChatMessageHistory

app = FastAPI()

# 跨域配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 聊天机器人
class ChatBot():

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.store = {}
        # 这是Qwen
        model = ChatOpenAI(
            api_key='sk-ad84e84254604e1aa98aa0b38715544b',
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model="qwen-plus",
        )
        # 这是GPT
        # model = ChatOpenAI(
        #     api_key='想用GPT去这里申请https://ai-yyds.com/about，或者去官网也行 ',
        #     base_url="https://ai-yyds.com/v1",
        #     model="gpt-3.5-turbo",
        # )



        prompt_with_history = ChatPromptTemplate.from_messages([
            # 按照我这里设置的提示词的样子你可以按照你的需求自定义你想要的效果
            ("system", "你将扮演一个机器人回答用户的选项。你看到结束的时候就停止角色扮演。 假设有三个选项分别是Economic impact、Public health impact、moral impact。如果你看到对应的选项的话你会做出相应的第一句回答,除了给定的内容不要返回其它的。第一句话过后你就可以自由发挥了。以下是第一句话应该回答的内容：\n 1.如果看到Economic impact，你会说：Hi ， can you tell me the economic impact of climate change？ \n 如果看到Public health你会说: Hi， can you tell me the public health impact of climate change？ \n 如果看到climate change你会说：Hi, can you tell me how does climate change contribute to inequality？ \n  "),
            MessagesPlaceholder(variable_name="history_msg"),
            ("human", "{query}")
        ])
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你的名字叫小智。你擅长对话，但是你不能说你是人工智能助手以及你的真实身份。"),
            ("human", "{query}")
        ])
        output_parser = StrOutputParser()

        self.chain = prompt | model | output_parser
        self.chain_with_history = RunnableWithMessageHistory(
            runnable= prompt_with_history | model | output_parser,
            get_session_history=self.get_history,
            input_messages_key='query',
            history_messages_key='history_msg',
            history_factory_config=[
                ConfigurableFieldSpec(
                    id="user_id",
                    annotation=str,
                    name="用户 ID",
                    description="用户的唯一标识符。",
                    default="",
                    is_shared=True,
                ),
                ConfigurableFieldSpec(
                    id="conversation_id",
                    annotation=str,
                    name="对话 ID",
                    description="对话的唯一标识符。",
                    default="",
                    is_shared=True,
                ),
            ],
        )



    def run_with_history(self, query: str, user_id: str, conversation_id: str):
        res = self.chain_with_history.invoke(
            input={"query": query},
            config={"configurable": {"user_id": user_id, "conversation_id": conversation_id}}
        )
        return res

    def run(self, query: str):
        res = self.chain.invoke(input={"query": query})
        return res
    # user_id和conversation_id共同标识全局唯一的聊天窗口
    def get_history(self, user_id: str, conversation_id: str) -> ChatMessageHistory:
        key = (user_id, conversation_id)
        if key not in self.store:
            self.store[key] = ChatMessageHistory()
        return self.store[key]



class Item(BaseModel):
    query: str
    chat_with_history: bool=True
    user_id: str=''
    conversation_id: str=''

chatBot = ChatBot()

#对外暴露的接口
@app.post('/chat')
def handle_query(item: Item):
    print(item)
    if item.chat_with_history:
        res = chatBot.run_with_history(item.query,user_id=item.user_id, conversation_id=item.conversation_id)
    else:
        res = chatBot.run(item.query)
    return {'response': res}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
