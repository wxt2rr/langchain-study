from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv(override=True)

model = init_chat_model(model="deepseek-chat", model_provider="deepseek")


class ContactInfo(BaseModel):
    """Contact information for a person."""
    name: str = Field(description="The name of the person")
    email: str = Field(description="The email address of the person")
    phone: str = Field(description="The phone number of the person")


def format_output():
    agent = create_agent(
        model=model,
        response_format=ContactInfo  # Auto-selects ProviderStrategy
    )

    result = agent.invoke({
        "messages": [
            {"role": "user", "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567"}]
    })

    print(result)
    print(result["structured_response"])
    # ContactInfo(name='John Doe', email='john@example.com', phone='(555) 123-4567')


if __name__ == '__main__':
    format_output()
