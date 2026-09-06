from fastmcp import FastMCP

mcp = FastMCP("AI Video Generator")


@mcp.tool
def create_video(prompt: str, duration: int = 5, aspect_ratio: str = "16:9") -> dict:
    """Create a video from a text prompt.

    This is currently a mock implementation used to test LLM tool calling through MCP.
    """

    return {
        "status": "queued",
        "job_id": "test-123",
        "prompt": prompt,
        "duration": duration,
        "aspect_ratio": aspect_ratio,
        "message": "Video generation request accepted.",
    }


@mcp.tool
def get_video_status(job_id: str) -> dict:
    """Get the current status of a video-generation job."""

    return {
        "job_id": job_id,
        "status": "completed",
        "message": "Mock video generation completed.",
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")