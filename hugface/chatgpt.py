from discord import Message
from redbot.core import Config, checks, commands
from typing import List
import openai
from openai import OpenAI

import re
base_url="https://api-inference.huggingface.co/v1/"

class ChatGPT(commands.Cog):
    """Send messages to HF"""

    def __init__(self, bot):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=359554929893)
        default_global = {
            "openai_api_key": None,
            "model": "Qwen/Qwen2.5-Coder-32B-Instruct",
            "max_tokens": 700,
            "mention": True,
            "reply": True
           
        }
        self.config.register_global(**default_global)
        self.client = None

    async def openai_api_key(self):
        openai_keys = await self.bot.get_shared_api_tokens("openai")
        openai_api_key = openai_keys.get("api_key")
        if openai_api_key is None:
            # Migrate key from config if exists
            openai_api_key = await self.config.openai_api_key()
            if openai_api_key is not None:
                await self.bot.set_shared_api_tokens("openai", api_key=openai_api_key)
                await self.config.openai_api_key.set(None)
        return openai_api_key

    @commands.Cog.listener()
    async def on_message(self, message: Message):
        if message.author.bot:
            return

        config_mention = await self.config.mention()
        config_reply = await self.config.reply()
        if not config_mention and not config_reply:
            return

        ctx: commands.Context = await self.bot.get_context(message)
        to_strip = f"(?m)^(<@!?{self.bot.user.id}>)"
        is_mention = config_mention and re.search(to_strip, message.content)
        is_reply = False
        if config_reply and message.reference and message.reference.resolved:
            author = getattr(message.reference.resolved, "author")
            if author is not None:
                is_reply = message.reference.resolved.author.id == self.bot.user.id and ctx.me in message.mentions
        if is_mention or is_reply:
            await self.do_chatgpt(ctx)

    @commands.command(aliases=['chat'])
    async def huggingface(self, ctx: commands.Context, *, message: str):
        """Send a message to HF."""
        await self.do_chatgpt(ctx, message)

    async def do_chatgpt(self, ctx: commands.Context, message: str = None):
        await ctx.typing()
        openai_api_key = await self.openai_api_key()
        if openai_api_key == None:
            prefix = ctx.prefix if ctx.prefix else "[p]"
            await ctx.send(f"HF API key not set. Use `{prefix}set api openai api_key <value>`.\nAn API key may be acquired from: huggingface.")
            return
        model = await self.config.model()
        if model == None:
            await ctx.send("HF model not set.")
            return
        max_tokens = await self.config.max_tokens()
        if max_tokens == None:
            await ctx.send("HF max_tokens not set.")
            return
        messages = []
        await self.build_messages(ctx, messages, ctx.message, message)
        reply = await self.call_api(
            model=model,
            api_key=openai_api_key,
            messages=messages,
            
            max_tokens=max_tokens
        )
        if len(reply) > 2000:
            reply = reply[:1997] + "..."
        await ctx.send(
            content=reply,
            reference=ctx.message
        )

    async def build_messages(self, ctx: commands.Context, messages: List[Message], message: Message, messageText: str = None):
        role = "assistant" if message.author.id == self.bot.user.id else "user"
        content = messageText if messageText else message.clean_content
        to_strip = f"(?m)^(<@!?{self.bot.user.id}>)"
        is_mention = re.search(to_strip, message.content)
        if is_mention:
            content = content[len(ctx.me.display_name) + 2 :]
        if role == "user" and content.startswith('chat '):
            content = content[5:]
        messages.insert(0, {"role": role, "content": content })
        if message.reference and message.reference.resolved:
            await self.build_messages(ctx, messages, message.reference.resolved)

    
    async def call_api(self, messages, model: str, api_key: str, max_tokens: int):
       try:
        if self.client is None:
         self.client = OpenAI(api_key=api_key, base_url=base_url)
        
        stream = self.client.chat.completions.create(
            model=model, 
            messages=messages, 
            temperature=0.8,
            max_tokens=max_tokens,
            top_p=0.5,
            stream=True
        )
        
        reply = []
        for chunk in stream:
         reply.append(chunk.choices[0].delta.content or "")
        reply = "".join(reply)

                
            
        if not reply:
                return "The message from model was empty."
        else:
                return reply
       except openai.APIConnectionError as e:
            return f"Failed to connect to HF API: {e}"
       except openai.RateLimitError as e:
            return f"HF API request exceeded rate limit: {e}"
       except openai.AuthenticationError as e:
            return f"HF API returned an Authentication Error: {e}"
       except openai.APIError as e:
            return f"HF API returned an API Error: {e}"

    @commands.command()
    @checks.is_owner()
    async def gethfmodel(self, ctx: commands.Context):
        """Get the model for HF.

        Defaults to "Qwen/Qwen2.5-Coder-32B-Instruct` See huggingface for a list of models."""
        model = await self.config.model()
        await ctx.send(f"HF model set to `{model}`")

    @commands.command()
    @checks.is_owner()
    async def sethfmodel(self, ctx: commands.Context, model: str):
        """Set the model for HF.

        Defaults to `Qwen/Qwen2.5-Coder-32B-Instruct` See huggingface for a list of models."""
        await self.config.model.set(model)
        await ctx.send("HF model set.")

    @commands.command()
    @checks.is_owner()
    async def gethftokens(self, ctx: commands.Context):
        """Get the maximum number of tokens for Model to generate.

        Defaults to `700`, see HF."""
        model = await self.config.max_tokens()
        await ctx.send(f"HF maximum number of tokens set to `{model}`")

    @commands.command()
    @checks.is_owner()
    async def sethftokens(self, ctx: commands.Context, number: str):
        """Set the maximum number of tokens for HF to generate.

        Defaults to `700` See HF."""
        try:
            await self.config.max_tokens.set(int(number))
            await ctx.send("HF maximum number of tokens set.")
        except ValueError:
            await ctx.send("Invalid numeric value for maximum number of tokens.")

    @commands.command()
    @checks.is_owner()
    async def togglehfmention(self, ctx: commands.Context):
        """Toggle messages to HF on mention.

        Defaults to `True`."""
        mention = not await self.config.mention()
        await self.config.mention.set(mention)
        if mention:
            await ctx.send("Enabled sending messages to HF on bot mention.")
        else:
            await ctx.send("Disabled sending messages to HF on bot mention.")

    @commands.command()
    @checks.is_owner()
    async def togglehfreply(self, ctx: commands.Context):
        """Toggle messages to HF on reply.

        Defaults to `True`."""
        reply = not await self.config.reply()
        await self.config.reply.set(reply)
        if reply:
            await ctx.send("Enabled sending messages to HF on bot reply.")
        else:
            await ctx.send("Disabled sending messages to HF on bot reply.")
