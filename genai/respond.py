import anthropic
import mlflow.sklearn
import sys
sys.path.append(".")
from ml.src.monitor import log_interaction

model = mlflow.sklearn.load_model("mlruns/1/models/m-8994ea6fe0c74a24af38f9be393ede0a/artifacts")


def classify_ticket(ticket_text):
    return model.predict([ticket_text])[0]



def draft_response(ticket_text, priority):
    client = anthropic.Anthropic()
    response = client.messages.create(
        model = "claude-sonnet-4-5",
        max_tokens=300,
        messages= [
            { "role": "user", "content": f"You are a customer support agent. This ticket has been classified as {priority}. Draft a professional response to: {ticket_text}" }
        ]
    )
    return response.content[0].text





def main():
    tickets = [
        "My internet is down and I have an important meeting in 30 minutes!",
        "I want to change my billing address.",
        "The app is crashing every time I try to open it."]
    for ticket in tickets:
        priority = classify_ticket(ticket)
        response = draft_response(ticket, priority)
        log_interaction(ticket, priority, response)
        print(f"Ticket: {ticket}\nPriority: {priority}\nDrafted Response: {response}\n{'-'*50}")


if __name__ == "__main__":
    main()

