from langchain.memory import ConversationBufferMemory
from langchain.agents import ConversationalChatAgent, AgentExecutor

class ChatAgent:
    def __init__(self, llm, tool_manager):
        self.llm = llm
        self.tool_manager = tool_manager
        self.memory = ConversationBufferMemory(memory_key="chat_history",input_key="input", return_messages=True)
        self.agent = ConversationalChatAgent.from_llm_and_tools(llm=self.llm, tools=list(self.tool_manager.tools.values()), system_message="You are a smart assistant whose main goal is to recommend amazing books and movies to users. Provide helpful, **short** and concise recommendations with a touch of fun!")
        self.chat_agent = AgentExecutor.from_agent_and_tools(agent=self.agent, tools=list(self.tool_manager.tools.values()), verbose=False, memory=self.memory)

    def get_response(self, query, topic_classifier):
        topic = topic_classifier.classify(query)
        tool_name = None if topic == "other" else topic.capitalize() + "Tool"

        try:
            response = self.chat_agent.run(input=query, tool_name=tool_name) if tool_name else self.llm.generate(prompt=query)
        except ValueError as e:
            response = str(e)

        return {"answer": response}