"""MCP Server."""
from mcp.server.fastmcp import FastMCP

from afrilingua.engine import build_engine

mcp = FastMCP("AfriLingua")
engine = build_engine()
...
