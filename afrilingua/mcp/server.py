"""MCP Server."""
from mcp.server.fastmcp import FastMCP

from afrilingua.engine import build_engine

mcp = FastMCP("AfriLingua")
engine = build_engine()

@mcp.tool()
def translate_text(text: str, tgt_lang: str) -> str:
    """Translate text to another language."""
    return engine.translate_one(text, None, tgt_lang).text

@mcp.resource("afrilingua://status")
def status():
    """Get AfriLingua engine status."""
    return str(engine.status())
