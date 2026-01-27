"""
Interactive Gemini AI Agent (Quota-Friendly)
Installation: pip install google-genai
"""

import os
import time
from google import genai
from google.genai import types


class InteractiveGeminiAgent:
    """An interactive AI agent that respects quota limits"""
    
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.model_name = 'gemini-2.5-flash'
        self.chat_session = None
        self.request_count = 0
        self.last_request_time = 0
    
    def _rate_limit_check(self):
        """Add delay to respect rate limits (15 req/min)"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        # Wait at least 4 seconds between requests (15 per minute = 1 per 4 sec)
        if time_since_last < 4:
            wait_time = 4 - time_since_last
            print(f"⏳ Rate limiting: waiting {wait_time:.1f}s...")
            time.sleep(wait_time)
        
        self.last_request_time = time.time()
        self.request_count += 1
    
    def generate(self, prompt: str, temperature: float = 0.7) -> str:
        """Generate text with rate limiting"""
        self._rate_limit_check()
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=500
                )
            )
            return response.text
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                return "⚠️ Quota exceeded. Please wait 1-2 minutes and try again."
            return f"Error: {str(e)}"
    
    def start_chat(self):
        """Start interactive chat mode"""
        self.chat_session = self.client.chats.create(model=self.model_name)
        print("\n" + "="*60)
        print("🤖 INTERACTIVE CHAT MODE")
        print("="*60)
        print("Commands:")
        print("  - Type your message to chat")
        print("  - Type 'quit' or 'exit' to end")
        print("  - Type 'clear' to start new conversation")
        print("="*60 + "\n")
    
    def chat_loop(self):
        """Run interactive chat loop"""
        self.start_chat()
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("👋 Goodbye!")
                    break
                
                if user_input.lower() == 'clear':
                    self.chat_session = self.client.chats.create(model=self.model_name)
                    print("🔄 Chat cleared! Starting fresh.\n")
                    continue
                
                self._rate_limit_check()
                
                try:
                    response = self.chat_session.send_message(user_input)
                    print(f"\n🤖 Agent: {response.text}\n")
                except Exception as e:
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                        print("\n⚠️ Quota exceeded. Waiting 60 seconds...\n")
                        time.sleep(60)
                        # Retry once
                        response = self.chat_session.send_message(user_input)
                        print(f"\n🤖 Agent: {response.text}\n")
                    else:
                        print(f"\n❌ Error: {str(e)}\n")
            
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except EOFError:
                print("\n\n👋 Goodbye!")
                break


def main():
    """Main function"""
    API_KEY = os.getenv('xxxxxx', 'xxxxxxxxx')
    
    if API_KEY == 'YOUR_API_KEY_HERE':
        print("❌ Please set your GEMINI_API_KEY!")
        print("\nOptions:")
        print("1. export GEMINI_API_KEY='your-key-here'")
        print("2. Edit this file and replace YOUR_API_KEY_HERE")
        return
    
    agent = InteractiveGeminiAgent(api_key=API_KEY)
    
    print("\n" + "="*60)
    print("🚀 GEMINI AI AGENT - INTERACTIVE MODE")
    print("="*60)
    print("\nThis agent includes automatic rate limiting to avoid quota errors.")
    print(f"Total requests this session: {agent.request_count}")
    print("\nStarting chat mode...\n")
    
    # Start interactive chat
    agent.chat_loop()
    
    print(f"\n📊 Session stats: {agent.request_count} requests made")


if __name__ == "__main__":
    main()