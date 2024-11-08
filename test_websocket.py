import asyncio
import websockets
import json

async def test_websocket():
    uri = "ws://localhost:8000/ws/test_user"
    async with websockets.connect(uri) as websocket:
        # Test question
        question = {
            "question": "What is the company's vacation policy?"
        }
        
        # Send question
        await websocket.send(json.dumps(question))
        print(f"Sent question: {question['question']}")
        
        # Get response
        response = await websocket.recv()
        print(f"Received response: {json.loads(response)}")

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(test_websocket())