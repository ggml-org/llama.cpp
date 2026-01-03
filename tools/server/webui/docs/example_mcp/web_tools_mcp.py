#!/usr/bin/env python3
"""
Web Tools MCP Server

Provides web search and web scraping capabilities via MCP protocol.

Tools:
- search_web: Search using SearxNG at localhost:8181
- scrape_website: Scrape websites using Selenium Grid at localhost:4444

Copyright 2025
"""

import sys
import logging
import asyncio
import httpx
from typing import Optional

# Redirect stdout to stderr before MCP imports (MCP uses stdout for protocol)
original_stdout = sys.stdout
sys.stdout = sys.stderr

from mcp.server import Server
from mcp.types import Tool, TextContent
from mcp import ServerSession, StdioServerParameters
import mcp.server.stdio

# Configure logging to stderr
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger("web_tools_mcp")

# Restore stdout for MCP protocol
sys.stdout = original_stdout

# Initialize MCP server
app = Server("web-tools")


# ============================================================================
# TOOLS
# ============================================================================

@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available web tools"""
    return [
        Tool(
            name="search_web",
            description=(
                "Search the web for current information using SearxNG.\n\n"
                "Use this to:\n"
                "- Find current information and facts\n"
                "- Research topics\n"
                "- Stay up-to-date with recent events\n"
                "- Gather information not in your training data\n\n"
                "Requires: SearxNG running at localhost:8181"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query string"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 5)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="scrape_website",
            description=(
                "Scrape a website to extract content using Selenium Grid.\n\n"
                "Use this to:\n"
                "- Extract information from specific websites\n"
                "- Gather detailed content from web pages\n"
                "- Analyze web content\n"
                "- Access dynamic web pages\n\n"
                "Requires: Selenium Grid running at localhost:4444"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Website URL to scrape (must include http:// or https://)"
                    },
                    "extract_text": {
                        "type": "boolean",
                        "description": "Whether to extract text content (default: true)",
                        "default": True
                    },
                    "extract_links": {
                        "type": "boolean",
                        "description": "Whether to extract links from the page (default: false)",
                        "default": False
                    }
                },
                "required": ["url"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls"""

    if name == "search_web":
        query = arguments["query"]
        limit = arguments.get("limit", 5)
        logger.info(f"search_web: query='{query}' limit={limit}")
        result = await _search_web(query=query, limit=limit)
        return [TextContent(type="text", text=result)]

    elif name == "scrape_website":
        url = arguments["url"]
        logger.info(f"scrape_website: url='{url}'")
        result = await _scrape_website(
            url=url,
            extract_text=arguments.get("extract_text", True),
            extract_links=arguments.get("extract_links", False)
        )
        return [TextContent(type="text", text=result)]

    else:
        raise ValueError(f"Unknown tool: {name}")


# ============================================================================
# TOOL IMPLEMENTATIONS
# ============================================================================

async def _search_web(query: str, limit: int = 5) -> str:
    """
    Search the web using SearxNG

    Args:
        query: Search query string
        limit: Maximum number of results

    Returns:
        Formatted search results or error message
    """
    try:
        url = "http://localhost:8181/search"
        params = {
            "q": query,
            "format": "json",
            "language": "en",
            "categories": "general",
            "safesearch": 1,
            "count": limit
        }

        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()

        data = response.json()
        results = data.get("results", [])[:limit]

        if not results:
            return f"No results found for: {query}"

        output = f"Search results for '{query}':\n\n"
        for i, r in enumerate(results, 1):
            output += f"{i}. {r.get('title', 'No title')}\n"
            output += f"   URL: {r.get('url', 'No URL')}\n"
            snippet = r.get('content', '')
            if snippet:
                output += f"   {snippet[:200]}...\n"
            output += "\n"

        return output

    except httpx.ConnectError:
        logger.error("Could not connect to SearxNG at localhost:8181")
        return "Error: SearxNG is not available. Make sure it's running at localhost:8181"
    except Exception as e:
        logger.error(f"search_web failed: {e}")
        return f"Error searching web: {str(e)}"


async def _scrape_website(
    url: str,
    extract_text: bool = True,
    extract_links: bool = False
) -> str:
    """
    Scrape a website using Selenium Grid

    Args:
        url: Website URL to scrape
        extract_text: Whether to extract text content
        extract_links: Whether to extract links

    Returns:
        Scraped content or error message
    """
    if not url.startswith(('http://', 'https://')):
        return "Error: Invalid URL. Must start with http:// or https://"

    try:
        # Check Selenium Grid availability
        grid_url = "http://localhost:4444/wd/hub"
        async with httpx.AsyncClient(timeout=10) as client:
            try:
                status_resp = await client.get(f"{grid_url}/status")
                if status_resp.status_code != 200:
                    return "Error: Selenium Grid is not available at localhost:4444"
            except:
                return "Error: Selenium Grid is not available at localhost:4444"

            # Create session
            capabilities = {
                "capabilities": {
                    "alwaysMatch": {
                        "browserName": "chrome",
                        "platformName": "ANY",
                        "goog:chromeOptions": {
                            "args": ["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu", "--headless"]
                        }
                    }
                }
            }

            session_resp = await client.post(
                f"{grid_url}/session",
                json=capabilities,
                headers={'Content-Type': 'application/json'},
                timeout=60
            )

            if session_resp.status_code != 200:
                return f"Error: Failed to create browser session"

            session_data = session_resp.json()
            session_id = session_data['value']['sessionId']

            try:
                # Navigate to URL
                await client.post(
                    f"{grid_url}/session/{session_id}/url",
                    json={"url": url},
                    headers={'Content-Type': 'application/json'},
                    timeout=60
                )

                # Wait a moment for page to load
                await asyncio.sleep(2)

                output = f"Scraped content from: {url}\n\n"

                # Extract text if requested
                if extract_text:
                    script = "return document.body.innerText || document.body.textContent || '';"
                    text_resp = await client.post(
                        f"{grid_url}/session/{session_id}/execute/sync",
                        json={"script": script, "args": []},
                        headers={'Content-Type': 'application/json'},
                        timeout=60
                    )

                    if text_resp.status_code == 200:
                        text_data = text_resp.json()
                        text = text_data['value']
                        # Truncate if too long
                        if len(text) > 5000:
                            text = text[:5000] + "... [truncated]"
                        output += f"Text Content:\n{text}\n\n"

                # Extract links if requested
                if extract_links:
                    link_script = """
                    const links = Array.from(document.querySelectorAll('a[href]'));
                    return links.map(link => ({
                        text: link.innerText || link.textContent || '',
                        href: link.href
                    })).filter(link => link.href && link.href.startsWith('http')).slice(0, 20);
                    """
                    link_resp = await client.post(
                        f"{grid_url}/session/{session_id}/execute/sync",
                        json={"script": link_script, "args": []},
                        headers={'Content-Type': 'application/json'},
                        timeout=60
                    )

                    if link_resp.status_code == 200:
                        link_data = link_resp.json()
                        links = link_data['value']
                        output += f"Links Found ({len(links)}):\n"
                        for i, link in enumerate(links[:10], 1):
                            output += f"{i}. {link.get('text', 'No text')[:50]} - {link.get('href')}\n"

                return output

            finally:
                # Clean up session
                try:
                    await client.delete(f"{grid_url}/session/{session_id}", timeout=10)
                except:
                    pass

    except httpx.ConnectError:
        logger.error("Could not connect to Selenium Grid at localhost:4444")
        return "Error: Selenium Grid is not available at localhost:4444"
    except Exception as e:
        logger.error(f"scrape_website failed: {e}")
        return f"Error scraping website: {str(e)}"


# ============================================================================
# MAIN
# ============================================================================

async def main():
    """Run the MCP server"""
    logger.info("Web Tools MCP Server starting...")
    logger.info("Tools: search_web, scrape_website")
    logger.info("Requirements:")
    logger.info("  - SearxNG at localhost:8181 (for search_web)")
    logger.info("  - Selenium Grid at localhost:4444 (for scrape_website)")

    # Run server with stdio transport
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
