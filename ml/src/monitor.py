import json, datetime, mlflow, os




def save_to_json(ticket_text, predicted_priority, drafted_response, actual_priority):
    os.makedirs("logs", exist_ok=True)
    
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "ticket_text": ticket_text,
        "predicted_priority": predicted_priority,
        "drafted_response": drafted_response,
        "actual_priority": actual_priority
    }
    
    with open("logs/interactions.json", "a") as f:
        f.write(json.dumps(entry) + "\n")

def log_interaction(ticket_text, predicted_priority, drafted_response, actual_priority = None):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("support-ticket-monitoring")
    with mlflow.start_run():
        mlflow.log_param("ticket_text", ticket_text)
        mlflow.log_param("predicted_priority", predicted_priority)
        mlflow.log_metric("response_length", len(drafted_response))
        if actual_priority:
            correct_prediction = 1.0 if predicted_priority == actual_priority else 0.0
            mlflow.log_metric("correct_prediction", correct_prediction)

    save_to_json(ticket_text, predicted_priority, drafted_response, actual_priority)
    print(f"Logged interaction — priority: {predicted_priority}, response length: {len(drafted_response)} chars")

