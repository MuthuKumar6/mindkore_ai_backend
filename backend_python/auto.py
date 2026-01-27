import google.generativeai as genai

genai.configure(api_key="AIzaSyDajFt0yA77gVv-op4gi-CRwlqRgdVkpos")

model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",  # or gemini-1.5-flash, gemini-3-flash, etc.
    tools=[{
        "function_declarations": [
            {
                "name": "send_email",
                "description": "Send an email to someone",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "to": {"type": "string", "description": "Recipient email"},
                        "subject": {"type": "string"},
                        "body": {"type": "string", "description": "Full email content"}
                    },
                    "required": ["to", "subject", "body"]
                }
            }
            # add more tools...
        ]
    }]
)

# Chat session (supports multi-turn)
chat = model.start_chat()

response = chat.send_message("Send reminder email to team@company.com about standup tomorrow 10 AM")

# Check if model wants to call a function
if response.candidates[0].content.parts[0].function_call:
    fc = response.candidates[0].content.parts[0].function_call
    if fc.name == "send_email":
        args = fc.args
        # Here YOU actually send the email (use smtplib, Gmail API, etc.)
        print(f"Sending email to {args['to']} ...")
        # ... your send logic ...
        # Then send result back
        chat.send_message(genai.protos.Part(
            function_response=genai.protos.FunctionResponse(
                name="send_email",
                response={"result": "Email sent successfully"}
            )
        ))