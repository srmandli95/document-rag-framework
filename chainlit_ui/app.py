import chainlit as cl 

from services.api_client import check_backend_health

@cl.on_chat_start
async def on_chat_start():

    await cl.Message(
        content=(
            "Welcome to the Personal Policy RAG Assistant! \n\n"
            "This assistant helps you retrieve information from your personal policies. \n\n"
            "To get started, simply upload your policy documents and ask any questions you have about them."
        )
    ).send()

    try:
        health_status = await check_backend_health()
        await cl.Message(
            content=(
                f"Status: {health_status['status']} \n\n"
                f"Service: {health_status['service']}   "
            )
        ).send()
    except Exception:
        cl.Message(
            content=(
                "Unable to connect to Personal Policy RAG Assistant backend. Please try again."
            )
        ).send()

@cl.on_message
async def on_message(message: cl.Message):
    await cl.Message(
        content=(
            "Day1: Received your message! This is a placeholder response. The backend integration is still in progress."
            )
        ).send()
            

        
    
        