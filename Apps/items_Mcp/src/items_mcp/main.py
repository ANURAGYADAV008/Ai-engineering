from fastmcp import FastMCP
from typing import List
from items_mcp.utils import retrieve_items_data, process_context
mcp=FastMCP("items_mcp_server")

@mcp.tool()
def get_formatted_context(query: str, top_k: int = 5) -> str:
    """
    Get the top k context, each representing an inventory item for a given query.

    Args:
        query: The query to get the top k context for.
        top_k: The number of context chunks to retrieve. Works best with 5 or more.

    Returns:
        A string containing the top k context chunks with IDs and average ratings
        prepended to each chunk, where each chunk represents an inventory item
        for the given query.
    """

    context = retrieve_items_data(query, top_k)
    formatted_context = process_context(context)

    return formatted_context

if __name__=="__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8000, path="/")